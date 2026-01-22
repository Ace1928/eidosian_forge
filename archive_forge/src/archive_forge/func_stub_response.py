import os
import unittest
from gae_ext_runtime import testutil
def stub_response(self, response):
    """Stubs the console response from the user.

        Args:
            response: (str) stubbed response.

        Returns:
            A function to reset the stubbed functions to their original
            implementations.
        """
    can_prompt = self.exec_env.CanPrompt
    prompt_response = self.exec_env.PromptResponse

    def unstub():
        self.exec_env.CanPrompt = can_prompt
        self.exec_env.PromptResponse = prompt_response
    self.exec_env.CanPrompt = lambda: True
    self.exec_env.PromptResponse = lambda prompt: response
    return unstub