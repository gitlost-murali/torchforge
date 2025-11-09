# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import uuid
from forge.envs.base import EnvState, EnvAction, EnvObservation


class EchoEnvironment:
    """
    A simple echo environment that returns the input messages unchanged.

    This environment is primarily used for testing and development purposes.
    It completes after a single step and returns a fixed reward of 1.0.
    """

    def __init__(self):
        self.env_state = EnvState(episode_id=str(uuid.uuid4()), step=0)
        self._reset_env_counter = 0
        self.max_steps = 1

    def reset(self) -> EnvObservation:
        """Reset the environment to initial state."""
        self.env_state = EnvState(episode_id=str(uuid.uuid4()), step=0)
        self._reset_env_counter += 1
        return EnvObservation(messages=[], reward=0.0, done=False)

    def step(self, action: EnvAction) -> EnvObservation:
        """
        Execute one step in the environment.

        Args:
            action: The action to take, containing messages

        Returns:
            EnvObservation with the same messages (echo), reward, and done status
        """
        self.env_state.step += 1
        stop_conditions = self.env_state.step >= self.max_steps
        return EnvObservation(
            messages=action.messages,  # No change since echo environment doesn't change the messages
            reward=1.0,
            done=stop_conditions,
        )
