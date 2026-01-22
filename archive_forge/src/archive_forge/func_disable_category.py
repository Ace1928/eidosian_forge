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
def disable_category(self, category: str, message_to_print: str) -> None:
    """Disable an entire category of commands.

        :param category: the category to disable
        :param message_to_print: what to print when anything in this category is run or help is called on it
                                 while disabled. The variable cmd2.COMMAND_NAME can be used as a placeholder for the name
                                 of the command being disabled.
                                 ex: message_to_print = f"{cmd2.COMMAND_NAME} is currently disabled"
        """
    all_commands = self.get_all_commands()
    for cmd_name in all_commands:
        func = self.cmd_func(cmd_name)
        if getattr(func, constants.CMD_ATTR_HELP_CATEGORY, None) == category:
            self.disable_command(cmd_name, message_to_print)