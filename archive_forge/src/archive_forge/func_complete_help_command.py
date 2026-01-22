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
def complete_help_command(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
    """Completes the command argument of help"""
    topics = set(self.get_help_topics())
    visible_commands = set(self.get_visible_commands())
    strs_to_match = list(topics | visible_commands)
    return self.basic_complete(text, line, begidx, endidx, strs_to_match)