import functools
import inspect
import json
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, cast
from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel
def get_fn_signature(fn: Callable):
    """Return the signature of a callable."""
    if not callable(fn):
        raise TypeError('The `source` filter only applies to callables.')
    source = textwrap.dedent(inspect.getsource(fn))
    re_search = re.search(re.compile('\\(([^)]+)\\)'), source)
    if re_search is None:
        signature = ''
    else:
        signature = re_search.group(1)
    return signature