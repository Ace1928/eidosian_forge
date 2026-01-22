import os
import subprocess
import sys
from .error import TryNext
def pre_prompt_hook(self):
    """ Run before displaying the next prompt

    Use this e.g. to display output from asynchronous operations (in order
    to not mess up text entry)
    """
    return None