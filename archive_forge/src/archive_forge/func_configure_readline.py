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
def configure_readline() -> None:
    """Configure readline tab completion and history"""
    nonlocal readline_configured
    nonlocal saved_completer
    nonlocal saved_history
    nonlocal parser
    if readline_configured:
        return
    if self._completion_supported():
        saved_completer = readline.get_completer()
        if completion_mode == utils.CompletionMode.NONE:

            def complete_none(text: str, state: int) -> Optional[str]:
                return None
            complete_func = complete_none
        elif completion_mode == utils.CompletionMode.COMMANDS:
            complete_func = self.complete
        else:
            if parser is None:
                parser = argparse_custom.DEFAULT_ARGUMENT_PARSER(add_help=False)
                parser.add_argument('arg', suppress_tab_hint=True, choices=choices, choices_provider=choices_provider, completer=completer)
            custom_settings = utils.CustomCompletionSettings(parser, preserve_quotes=preserve_quotes)
            complete_func = functools.partial(self.complete, custom_settings=custom_settings)
        readline.set_completer(complete_func)
    if completion_mode != utils.CompletionMode.COMMANDS or history is not None:
        saved_history = []
        for i in range(1, readline.get_current_history_length() + 1):
            saved_history.append(readline.get_history_item(i))
        readline.clear_history()
        if history is not None:
            for item in history:
                readline.add_history(item)
    readline_configured = True