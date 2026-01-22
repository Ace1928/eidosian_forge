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
@with_argparser(relative_run_script_parser)
def do__relative_run_script(self, args: argparse.Namespace) -> Optional[bool]:
    """
        Run commands in script file that is encoded as either ASCII or UTF-8 text

        :return: True if running of commands should stop
        """
    file_path = args.file_path
    relative_path = os.path.join(self._current_script_dir or '', file_path)
    return self.do_run_script(utils.quote_string(relative_path))