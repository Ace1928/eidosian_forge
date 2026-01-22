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
def register_postloop_hook(self, func: Callable[[], None]) -> None:
    """Register a function to be called at the end of the command loop."""
    self._validate_prepostloop_callable(func)
    self._postloop_hooks.append(func)