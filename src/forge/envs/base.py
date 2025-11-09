# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pydantic import BaseModel


class EnvState(BaseModel):
    """State of the environment."""
    episode_id: str
    step: int


class EnvAction(BaseModel):
    """Action taken in the environment."""
    messages: list[dict]


class EnvObservation(BaseModel):
    """Observation returned by the environment."""
    messages: list[dict]
    reward: float
    done: bool
