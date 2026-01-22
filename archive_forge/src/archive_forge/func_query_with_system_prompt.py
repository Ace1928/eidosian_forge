from __future__ import annotations
import logging
import time
from abc import ABC, abstractmethod
from typing import Callable
import openai
from typing_extensions import override
def query_with_system_prompt(self, system_prompt: str, prompt: str) -> str:
    """
        Abstract method to query an LLM with a given prompt and system prompt and return the response.

        Args:
            system prompt (str): The system prompt to send to the LLM.
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The response from the LLM.
        """
    return self.query(system_prompt + '\n' + prompt)