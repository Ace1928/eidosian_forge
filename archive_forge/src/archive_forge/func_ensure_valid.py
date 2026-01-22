import functools
import operator
import itertools
from .errors import OptionError
from .extern.jaraco.text import yield_lines
from .extern.jaraco.functools import pass_none
from ._importlib import metadata
from ._itertools import ensure_unique
from .extern.more_itertools import consume
def ensure_valid(ep):
    """
    Exercise one of the dynamic properties to trigger
    the pattern match.
    """
    try:
        ep.extras
    except AttributeError as ex:
        msg = f'Problems to parse {ep}.\nPlease ensure entry-point follows the spec: https://packaging.python.org/en/latest/specifications/entry-points/'
        raise OptionError(msg) from ex