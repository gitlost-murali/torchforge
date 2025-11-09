import torch
from forge.data_models.prompt import Prompt, Message, Role
import pytest
from unittest.mock import AsyncMock
from forge.envs import EchoEnvironment
from forge.data_models.completion import Completion
from apps.grpo.main_with_env import rollout_single_trajectory


@pytest.fixture
def mock_policy():
    """Create a mock policy for testing."""
    policy = AsyncMock()
    # Mock the generate.route method to return a completion
    policy.generate.route.return_value = [Completion(
        prompt=Prompt(messages=[Message(chunks=["Follow-up response"], role=Role.ASSISTANT)]),
        text="Follow-up response",
        prompt_ids=torch.tensor([7, 8, 9]),
        token_ids=torch.tensor([10, 11, 12]),
        logprobs=torch.tensor([0.4, 0.5, 0.6])
    )]
    return policy


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = AsyncMock()
    # Mock apply_chat_template to return a formatted string
    tokenizer.apply_chat_template.return_value = "formatted_chat_template"
    return tokenizer


@pytest.fixture
def initial_completion():
    """Create an initial completion for testing."""
    return Completion(
        prompt=Prompt(messages=[Message(chunks=["How are you?"], role=Role.USER)]),
        text="I am good",
        prompt_ids=torch.tensor([1, 2, 3]),
        token_ids=torch.tensor([4, 5, 6]),
        logprobs=torch.tensor([0.1, 0.2, 0.3])
    )


@pytest.mark.asyncio
async def test_rollout_single_trajectory_with_echo_environment(
    mock_policy,
    mock_tokenizer,
    initial_completion
):
    """Test the integration of the environment with the policy.

    EchoEnvironment completes after a single step, so the rollout should:
    1. Create the environment
    2. Execute one step with initial messages
    3. Get done=True from the environment
    4. Return the initial completion without calling policy.generate
    """
    system_prompt = "You are a helpful assistant."
    raw_question = "How are you?"

    # Run the rollout
    result = await rollout_single_trajectory(
        initial_response=initial_completion,
        policy=mock_policy,
        tokenizer=mock_tokenizer,
        system_prompt=system_prompt,
        raw_question=raw_question
    )

    # Verify the result is the initial completion since EchoEnvironment
    # completes in one step (done=True immediately)
    assert result == initial_completion

    # Verify that policy.generate.route was NOT called because the environment
    # completed before the while loop could execute
    mock_policy.generate.route.assert_not_called()
