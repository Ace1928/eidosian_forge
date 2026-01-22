import functools
import operator
import itertools
from .errors import OptionError
from .extern.jaraco.text import yield_lines
from .extern.jaraco.functools import pass_none
from ._importlib import metadata
from ._itertools import ensure_unique
from .extern.more_itertools import consume
def load_group(value, group):
    """
    Given a value of an entry point or series of entry points,
    return each as an EntryPoint.
    """
    lines = yield_lines(value)
    text = f'[{group}]\n' + '\n'.join(lines)
    return metadata.EntryPoints._from_text(text)