import argparse
import cmd
import functools
import glob
import inspect
import os
import pydoc
import re
import sys
import threading
from code import (
from collections import (
from contextlib import (
from types import (
from typing import (
from . import (
from .argparse_custom import (
from .clipboard import (
from .command_definition import (
from .constants import (
from .decorators import (
from .exceptions import (
from .history import (
from .parsing import (
from .rl_utils import (
from .table_creator import (
from .utils import (
def async_update_prompt(self, new_prompt: str) -> None:
    """
        Update the command line prompt while the user is still typing at it. This is good for alerting the user to
        system changes dynamically in between commands. For instance you could alter the color of the prompt to
        indicate a system status or increase a counter to report an event. If you do alter the actual text of the
        prompt, it is best to keep the prompt the same width as what's on screen. Otherwise the user's input text will
        be shifted and the update will not be seamless.

        IMPORTANT: This function will not update the prompt unless it can acquire self.terminal_lock to ensure
                   a prompt is onscreen. Therefore, it is best to acquire the lock before calling this function
                   to guarantee the prompt changes and to avoid raising a RuntimeError.

                   This function is only needed when you need to update the prompt while the main thread is blocking
                   at the prompt. Therefore, this should never be called from the main thread. Doing so will
                   raise a RuntimeError.

                   If user is at a continuation prompt while entering a multiline command, the onscreen prompt will
                   not change. However, self.prompt will still be updated and display immediately after the multiline
                   line command completes.

        :param new_prompt: what to change the prompt to
        :raises RuntimeError: if called from the main thread.
        :raises RuntimeError: if called while another thread holds `terminal_lock`
        """
    self.async_alert('', new_prompt)