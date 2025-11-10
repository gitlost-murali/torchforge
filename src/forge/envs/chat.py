# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import uuid
import json
from typing import Optional, Any
from forge.envs.base import EnvState, EnvAction, EnvObservation


CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a mathematical expression and return the result",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '2 + 2 * 3')",
                }
            },
            "required": ["expression"],
        },
    },
}

FINAL_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "final_answer",
        "description": "Provide the final answer to complete the task",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer to the question or task",
                }
            },
            "required": ["answer"],
        },
    },
}


class ChatEnvironment:
    """
    A chat environment that supports OpenAI-compatible tool calling.

    The environment provides calculator and final_answer tools that can be used
    with any OpenAI-compatible inference engine (OpenAI API, vLLM, etc.).

    Tools:
    - calculator: Evaluates mathematical expressions
    - final_answer: Provides the final answer and completes the episode

    The environment expects messages to follow OpenAI's message format with tool_calls.
    """

    def __init__(self, max_steps: int = 10):
        """
        Initialize the chat environment.

        Args:
            max_steps: Maximum number of steps before the episode terminates
        """
        self.env_state = EnvState(episode_id=str(uuid.uuid4()), step=0)
        self._reset_env_counter = 0
        self.max_steps = max_steps
        self._final_answer: Optional[str] = None
        self.tools = [CALCULATOR_TOOL, FINAL_ANSWER_TOOL]

    def reset(self) -> EnvObservation:
        """Reset the environment to initial state."""
        self.env_state = EnvState(episode_id=str(uuid.uuid4()), step=0)
        self._reset_env_counter += 1
        self._final_answer = None
        return EnvObservation(messages=[], reward=0.0, done=False)

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Get the list of available tools in OpenAI-compatible format.

        Returns:
            List of tool definitions that can be passed to the LLM
        """
        return self.tools

    def _execute_calculator(self, expression: str) -> str:
        """
        Execute a calculator operation.

        Args:
            expression: Mathematical expression to evaluate

        Returns:
            String result of the calculation or error message
        """
        try:
            # Clean the expression
            expression = expression.strip()
            # Safely evaluate mathematical expressions
            # Using eval with restricted globals/locals for safety
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        Execute a tool call.

        Args:
            tool_name: Name of the tool to execute
            arguments: Dictionary of arguments for the tool

        Returns:
            String result of the tool execution
        """
        if tool_name == "calculator":
            expression = arguments.get("expression", "")
            return self._execute_calculator(expression)
        elif tool_name == "final_answer":
            answer = arguments.get("answer", "")
            self._final_answer = answer
            return "Task completed successfully."
        else:
            return f"Error: Unknown tool '{tool_name}'"

    def step(self, action: EnvAction) -> EnvObservation:
        """
        Execute one step in the environment.

        Expects action.messages to contain OpenAI-format messages, where assistant
        messages may have 'tool_calls' containing tool invocations.

        Args:
            action: The action to take, containing messages

        Returns:
            EnvObservation with tool responses, reward, and done status
        """
        self.env_state.step += 1

        # Get the last message (assumed to be from the assistant)
        if not action.messages:
            return EnvObservation(
                messages=[],
                reward=0.0,
                done=self.env_state.step >= self.max_steps,
            )

        last_message = action.messages[-1]

        # Check if the message contains tool calls
        tool_calls = last_message.get("tool_calls", [])

        response_messages = []
        done = False
        reward = 0.0

        for tool_call in tool_calls:
            tool_id = tool_call.get("id", "")
            function = tool_call.get("function", {})
            tool_name = function.get("name", "")

            # Parse arguments (may be string or dict)
            arguments = function.get("arguments", "{}")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            # Execute the tool
            result = self._execute_tool(tool_name, arguments)

            # Create tool response message in OpenAI format
            response_messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": result,
            })

            # Check if this was a final_answer call
            if tool_name == "final_answer":
                done = True
                reward = 1.0

        # Check if max steps reached
        if self.env_state.step >= self.max_steps:
            done = True
            reward = 0.0

        return EnvObservation(
            messages=response_messages,
            reward=reward,
            done=done,
        )
