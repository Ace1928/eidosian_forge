import contextlib
import functools
import re
import textwrap
from typing import Iterable, List, Sequence, Tuple, Type
from typing_extensions import Literal, get_args, get_origin
from . import _resolver
def get_delimeter() -> Literal['-', '_']:
    """Get delimeter used to separate words."""
    return DELIMETER