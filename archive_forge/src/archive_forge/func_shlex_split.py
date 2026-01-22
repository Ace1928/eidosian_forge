import re
import shlex
from typing import (
import attr
from . import (
from .exceptions import (
def shlex_split(str_to_split: str) -> List[str]:
    """
    A wrapper around shlex.split() that uses cmd2's preferred arguments.
    This allows other classes to easily call split() the same way StatementParser does.

    :param str_to_split: the string being split
    :return: A list of tokens
    """
    return shlex.split(str_to_split, comments=False, posix=False)