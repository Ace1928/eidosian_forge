import contextlib
import functools
import re
import textwrap
from typing import Iterable, List, Sequence, Tuple, Type
from typing_extensions import Literal, get_args, get_origin
from . import _resolver
def strip_ansi_sequences(x: str):
    return _get_ansi_pattern().sub('', x)