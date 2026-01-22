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
def pwarning(self, msg: Any='', *, end: str='\n', apply_style: bool=True) -> None:
    """Wraps perror, but applies ansi.style_warning by default

        :param msg: object to print
        :param end: string appended after the end of the message, default a newline
        :param apply_style: If True, then ansi.style_warning will be applied to the message text. Set to False in cases
                            where the message text already has the desired style. Defaults to True.
        """
    if apply_style:
        msg = ansi.style_warning(msg)
    self.perror(msg, end=end, apply_style=False)