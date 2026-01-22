import functools
import inspect
import json
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, cast
from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel
def get_fn_description(fn: Callable):
    """Returns the first line of a callable's docstring."""
    if not callable(fn):
        raise TypeError('The `description` filter only applies to callables.')
    docstring = inspect.getdoc(fn)
    if docstring is None:
        description = ''
    else:
        description = docstring.split('\n')[0].strip()
    return description