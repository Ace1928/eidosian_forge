import functools
import inspect
import json
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, cast
from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel
def get_fn_source(fn: Callable):
    """Return the source code of a callable."""
    if not callable(fn):
        raise TypeError('The `source` filter only applies to callables.')
    source = textwrap.dedent(inspect.getsource(fn))
    re_search = re.search(re.compile('(\\bdef\\b.*)', re.DOTALL), source)
    if re_search is not None:
        source = re_search.group(0)
    else:
        raise TypeError("Could not read the function's source code")
    return source