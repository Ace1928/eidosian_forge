import functools
import operator
import itertools
from .errors import OptionError
from .extern.jaraco.text import yield_lines
from .extern.jaraco.functools import pass_none
from ._importlib import metadata
from ._itertools import ensure_unique
from .extern.more_itertools import consume
def render_items(eps):
    return '\n'.join((f'{ep.name} = {ep.value}' for ep in sorted(eps)))