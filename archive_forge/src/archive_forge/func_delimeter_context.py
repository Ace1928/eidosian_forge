import contextlib
import functools
import re
import textwrap
from typing import Iterable, List, Sequence, Tuple, Type
from typing_extensions import Literal, get_args, get_origin
from . import _resolver
@contextlib.contextmanager
def delimeter_context(delimeter: Literal['-', '_']):
    """Context for setting the delimeter. Determines if `field_a` is populated as
    `--field-a` or `--field_a`. Not thread-safe."""
    global DELIMETER
    delimeter_restore = DELIMETER
    DELIMETER = delimeter
    yield
    DELIMETER = delimeter_restore