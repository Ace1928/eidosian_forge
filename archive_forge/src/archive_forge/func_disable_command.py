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
def disable_command(self, command: str, message_to_print: str) -> None:
    """
        Disable a command and overwrite its functions

        :param command: the command being disabled
        :param message_to_print: what to print when this command is run or help is called on it while disabled

                                 The variable cmd2.COMMAND_NAME can be used as a placeholder for the name of the
                                 command being disabled.
                                 ex: message_to_print = f"{cmd2.COMMAND_NAME} is currently disabled"
        """
    if command in self.disabled_commands:
        return
    command_function = self.cmd_func(command)
    if command_function is None:
        raise AttributeError(f"'{command}' does not refer to a command")
    help_func_name = constants.HELP_FUNC_PREFIX + command
    completer_func_name = constants.COMPLETER_FUNC_PREFIX + command
    self.disabled_commands[command] = DisabledCommand(command_function=command_function, help_function=getattr(self, help_func_name, None), completer_function=getattr(self, completer_func_name, None))
    new_func = functools.partial(self._report_disabled_command_usage, message_to_print=message_to_print.replace(constants.COMMAND_NAME, command))
    setattr(self, self._cmd_func_name(command), new_func)
    setattr(self, help_func_name, new_func)
    setattr(self, completer_func_name, lambda *args, **kwargs: [])