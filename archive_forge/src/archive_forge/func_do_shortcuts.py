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
@with_argparser(shortcuts_parser)
def do_shortcuts(self, _: argparse.Namespace) -> None:
    """List available shortcuts"""
    sorted_shortcuts = sorted(self.statement_parser.shortcuts, key=lambda x: self.default_sort_key(x[0]))
    result = '\n'.join(('{}: {}'.format(sc[0], sc[1]) for sc in sorted_shortcuts))
    self.poutput(f'Shortcuts for other commands:\n{result}')
    self.last_result = True