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
def runcmds_plus_hooks(self, cmds: Union[List[HistoryItem], List[str]], *, add_to_history: bool=True, stop_on_keyboard_interrupt: bool=False) -> bool:
    """
        Used when commands are being run in an automated fashion like text scripts or history replays.
        The prompt and command line for each command will be printed if echo is True.

        :param cmds: commands to run
        :param add_to_history: If True, then add these commands to history. Defaults to True.
        :param stop_on_keyboard_interrupt: if True, then stop running contents of cmds if Ctrl-C is pressed instead of moving
                                           to the next command in the list. This is used when the commands are part of a
                                           group, like a text script, which should stop upon Ctrl-C. Defaults to False.
        :return: True if running of commands should stop
        """
    for line in cmds:
        if isinstance(line, HistoryItem):
            line = line.raw
        if self.echo:
            self.poutput(f'{self.prompt}{line}')
        try:
            if self.onecmd_plus_hooks(line, add_to_history=add_to_history, raise_keyboard_interrupt=stop_on_keyboard_interrupt):
                return True
        except KeyboardInterrupt as ex:
            if stop_on_keyboard_interrupt:
                self.perror(ex)
                break
    return False