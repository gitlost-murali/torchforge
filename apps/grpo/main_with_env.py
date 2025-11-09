# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage: python -m apps.grpo.main --config apps/grpo/qwen3_1_7b.yaml

import asyncio
import uuid

from datasets import load_dataset
from pydantic import BaseModel
import torch
import torchstore as ts
from forge.actors.reference_model import ReferenceModel
from forge.actors.replay_buffer import ReplayBuffer
from forge.actors.trainer import RLTrainer
from forge.controller.actor import ForgeActor
from forge.controller.provisioner import init_provisioner, shutdown
from forge.data.rewards import MathReward, ThinkingReward
from forge.data_models.completion import Completion
from forge.observability.metric_actors import get_or_create_metric_logger
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer

from monarch.actor import endpoint
from forge.types import LauncherConfig, ProvisionerConfig
from forge.util.config import parse
from omegaconf import DictConfig

from dataclasses import dataclass
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_tokenizer
from apps.grpo.main import Policy, ComputeAdvantages, \
                RewardActor, Episode, simple_grpo_loss, \
                collate, drop_weights

@dataclass
class DatasetActor(ForgeActor):
    """Actor wrapper for HuggingFace dataset to provide async interface."""

    path: str = "openai/gsm8k"
    revision: str = "main"
    data_split: str = "train"
    streaming: bool = True
    model: str = "Qwen/Qwen3-1.7B"

    @endpoint
    def setup(self):
        self._tokenizer = get_tokenizer(self.model)
        self._epoch = 0

        def gsm8k_transform(sample):
            system_prompt = """
            Put all your scratchpad work between <think> and </think> tags.
            Your final answer should be between <answer> and </answer> tags otherwise it will not be scored.
            """
            request: str = sample["question"]
            as_chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request},
            ]
            formatted_request = self._tokenizer.apply_chat_template(
                as_chat,
                tokenize=False,
                add_generation_prompt=True,
            )
            target: str = sample["answer"]
            formatted_target = target.split("#### ")[1]
            return {
                    "request": formatted_request, 
                    "target": formatted_target,
                    "raw_question": request,
                    "system_prompt": system_prompt.strip()
                    }

        self._base_dataset = load_dataset(
            self.path, self.revision, split=self.data_split, streaming=self.streaming
        )
        self._base_dataset = self._base_dataset.map(gsm8k_transform)
        self._base_dataset = self._base_dataset.shuffle()
        self._iterator = iter(self._base_dataset)

    @endpoint
    async def sample(self) -> dict[str, str] | None:
        try:
            sample = next(self._iterator)

            record_metric("dataset/sample/count_samples_generated", 1, Reduce.SUM)
            record_metric(
                "dataset/sample/avg_sample_len",
                len(sample["request"]),
                Reduce.MEAN,
            )
            record_metric("dataset/sample/current_epoch", self._epoch, Reduce.MAX)

            return sample
        except StopIteration:
            # Restart iterator for next epoch with reshuffling
            self._epoch += 1
            print(
                f"Dataset epoch {self._epoch - 1} completed. Starting epoch {self._epoch}"
            )
            self._base_dataset.set_epoch(self._epoch)
            self._iterator = iter(self._base_dataset)
            return next(self._iterator)

    @endpoint
    async def pad_token(self):
        return self._tokenizer.pad_token_id

    @endpoint
    async def get_tokenizer(self):
        return self._tokenizer



class EnvState(BaseModel):
    episode_id: str
    step: int

class EnvAction(BaseModel):
    messages: list[dict]

class EnvObservation(BaseModel):
    messages: list[dict]
    reward: float
    done: bool

class EchoEnvironment():
    def __init__(self):
        self.env_state = EnvState(episode_id=str(uuid.uuid4()), step=0)
        self._reset_env_counter = 0
        self.max_steps = 1

    def reset(self) -> EnvObservation:
        self.env_state = EnvState(episode_id=str(uuid.uuid4()), step=0)
        self._reset_env_counter += 1
        return EnvObservation(messages=[], reward=0.0, done=False)

    def step(self, action: EnvAction) -> EnvObservation:
        self.env_state.step += 1
        stop_conditions = self.env_state.step >= self.max_steps
        return EnvObservation(
            messages=action.messages, # No change since echo environment doesn't change the messages
            reward=1.0,
            done=stop_conditions,
        )



async def rollout_single_trajectory(
    initial_response: Completion,
    policy: Policy,
    tokenizer: AnyTokenizer,
    system_prompt: str,
    raw_question: str,
) -> Completion:
    """
    Rollout a single trajectory through the environment starting from an initial response.

    Returns the final completed response after environment interaction.
    """
    env = EchoEnvironment()

    chat_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": raw_question},
        {"role": "assistant", "content": initial_response.text}
    ]


    # Get first observation
    env_action = EnvAction(messages=chat_messages)
    env_observation = env.step(env_action)
    current_messages, reward, done = env_observation.messages, env_observation.reward, env_observation.done

    # Continue interaction until environment signals done
    latest_response = initial_response
    while not done:
        # Format messages for policy
        flattened_messages = tokenizer.apply_chat_template(
            current_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate next response from policy
        new_responses: list[Completion] = await policy.generate.route(flattened_messages)
        latest_response = new_responses[0]  # Take first completion

        # Add to conversation
        current_messages.append({"role": "assistant", "content": latest_response.text})

        # Step environment
        env_action = EnvAction(messages=current_messages)
        env_observation = env.step(env_action)
        current_messages, reward, done = env_observation.messages, env_observation.reward, env_observation.done

    return latest_response


async def main(cfg: DictConfig):
    """Main GRPO training loop with rollout and training processes."""
    group_size = cfg.group_size
    max_req_tokens = cfg.max_req_tokens
    max_res_tokens = cfg.max_res_tokens

    # ---- Global setups ---- #
    provisioner = None
    if cfg.get("provisioner", None) is not None:
        provisioner = await init_provisioner(
            ProvisionerConfig(launcher_config=LauncherConfig(**cfg.provisioner))
        )
    else:
        provisioner = await init_provisioner()

    metric_logging_cfg = cfg.get("metric_logging", {})
    mlogger = await get_or_create_metric_logger(process_name="Controller")
    await mlogger.init_backends.call_one(metric_logging_cfg)

    # ---- Setup services ---- #

    (
        dataloader,
        policy,
        trainer,
        replay_buffer,
        compute_advantages,
        ref_model,
        reward_actor,
    ) = await asyncio.gather(
        DatasetActor.options(**cfg.actors.dataset).as_actor(**cfg.dataset),
        Policy.options(**cfg.services.policy).as_service(**cfg.policy),
        RLTrainer.options(**cfg.actors.trainer).as_actor(
            **cfg.trainer, loss=simple_grpo_loss
        ),
        ReplayBuffer.options(**cfg.actors.replay_buffer).as_actor(
            **cfg.replay_buffer, collate=collate
        ),
        ComputeAdvantages.options(**cfg.actors.compute_advantages).as_actor(),
        ReferenceModel.options(**cfg.services.ref_model).as_service(**cfg.ref_model),
        RewardActor.options(**cfg.services.reward_actor).as_service(
            reward_functions=[MathReward(), ThinkingReward()]
        ),
    )

    # Set max_steps to the configured value, or -1 if not specified or Null
    max_steps = cfg.trainer.training.steps or -1

    print("All services initialized successfully!")
    shutdown_event = asyncio.Event()
    # Here we spawn a torchstore storage volume per trainer process.
    # We initialize after service initialization because torchstore currently
    # requires access to the underlying proc meshes in the local rank strategy.
    # We should be able to hide this in the future.
    # TODO: support multiple host meshes
    trainer_num_procs = cfg.actors.trainer["procs"]
    trainer_host_mesh_name = cfg.actors.trainer["mesh_name"]
    trainer_hosts = provisioner.get_host_mesh(trainer_host_mesh_name)
    await ts.initialize(
        mesh=trainer_hosts.spawn_procs(per_host={"procs": trainer_num_procs}),
        strategy=ts.LocalRankStrategy(),
    )
    print("Torchstore successfully initialized with local rank strategy")

    # ---- Core RL loops ---- #
    async def continuous_rollouts():
        rollout_count = 0
        pad_id = await dataloader.pad_token.call_one()
        env = EchoEnvironment()  # Create environment instance directly
        while not shutdown_event.is_set():
            t = Tracer("main_perf/continuous_rollouts")
            t.start()
            sample = await dataloader.sample.call_one()
            if sample is None:
                print("Dataloader is empty, exiting continuous rollout")
                return

            t.step("data_loading")

            prompt, target = sample["request"], sample["target"]
            raw_question, system_prompt = sample["raw_question"], sample["system_prompt"]
            responses: list[Completion] = await policy.generate.route(prompt)
            t.step("policy_generation")

            completed_trajectories = []

            tokenizer: AnyTokenizer = await dataloader.get_tokenizer.call_one()

            # Rollout with environment interaction
            for response in responses:
                completed_trajectory = await rollout_single_trajectory(
                    response, policy, tokenizer, system_prompt, raw_question
                )
                completed_trajectories.append(completed_trajectory)

            episodes = []
            # Construct episodes and calculate rewards
            input_ids = torch.ones(
                (group_size, max_req_tokens + max_res_tokens),
                dtype=torch.long,
            )
            for i, completed_trajectory in enumerate(completed_trajectories):
                episode = Episode(
                    episode_id=str(uuid.uuid4()),
                    pad_id=pad_id,
                    request_len=max_req_tokens,
                    response_len=max_res_tokens,
                    target=target,
                    completion=completed_trajectory,
                )
                episode.reward = await reward_actor.evaluate_response.route(
                    prompt=prompt, response=completed_trajectory.text, target=target
                )
                episodes.append(episode)

                # Build input_ids for reference logprobs
                input_ids[i, :max_req_tokens] = episode.request_tensor
                input_ids[i, max_req_tokens:] = episode.response_tensor

            t.step("reward_evaluation")

            ref_logprobs = await ref_model.forward.route(
                input_ids, max_req_tokens, return_logprobs=True
            )
            t.step("reference_model_calculate_logprobs")

            for i, episode in enumerate(episodes):
                episode.ref_logprobs = ref_logprobs[i]
            del ref_logprobs, input_ids

            advantages = await compute_advantages.compute.call_one(episodes)
            for episode, advantage in zip(episodes, advantages):
                episode.advantage = advantage
                await replay_buffer.add.call_one(episode)

            rollout_count += 1
            record_metric(
                "main/continuous_rollouts/count_rollout_iterations", 1, Reduce.SUM
            )
            t.stop()

    async def continuous_training():
        training_step = 0
        restart_tracer = True  # Flag to control when to restart tracer

        while max_steps == -1 or training_step < max_steps:
            # Restart tracer when needed (initial start or after completing a training step)
            # Otherwise, we cannot measure time waiting for buffer
            if restart_tracer:
                t = Tracer("main_perf/continuous_training")
                t.start()
                restart_tracer = False

            batch = await replay_buffer.sample.call_one(
                curr_policy_version=training_step
            )
            if batch is None:
                await asyncio.sleep(0.1)
            else:
                t.step("waiting_for_buffer")

                inputs, targets = batch
                await trainer.train_step.call(inputs, targets)
                training_step += 1
                t.step("train_step")

                await trainer.push_weights.call(training_step)
                t.step("push_weights")

                await policy.update_weights.fanout(training_step)
                t.step("update_weights")

                if training_step >= 2:
                    await drop_weights(training_step - 1)
                    t.step("drop_weights")

                t.stop()
                restart_tracer = True

                # Flush metrics every training step to WandB
                await mlogger.flush.call_one(training_step)

        print(
            f"Reached training limit ({max_steps} steps). Exiting continuous_training loop."
        )

    num_rollout_threads = cfg.get("rollout_threads", 1)
    num_training_threads = cfg.get("training_threads", 1)
    print(
        f"Starting GRPO with {num_rollout_threads} rollout threads, {num_training_threads} training threads"
    )
    rollout_tasks = [
        asyncio.create_task(continuous_rollouts()) for _ in range(num_rollout_threads)
    ]
    training_task = asyncio.create_task(continuous_training())

    try:
        await training_task
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        print("Shutting down... (this may take a few seconds)")
        shutdown_event.set()

        try:
            # Give rollouts up to 5s to finish naturally
            await asyncio.wait_for(
                asyncio.gather(*rollout_tasks, return_exceptions=True),
                timeout=5,
            )
        except asyncio.TimeoutError:
            print("Timeout waiting for rollouts; forcing cancellation...")
            for t in rollout_tasks:
                t.cancel()
            await asyncio.gather(*rollout_tasks, return_exceptions=True)

        training_task.cancel()

        await shutdown()


if __name__ == "__main__":

    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()  # @parse grabs the cfg from CLI
