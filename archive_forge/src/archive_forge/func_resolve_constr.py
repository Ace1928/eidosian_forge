import contextlib
import datetime
import ipaddress
import json
import math
from fractions import Fraction
from typing import Callable, Dict, Type, Union, cast, overload
import hypothesis.strategies as st
import pydantic
import pydantic.color
import pydantic.types
from pydantic.utils import lenient_issubclass
@resolves(pydantic.ConstrainedStr)
def resolve_constr(cls):
    min_size = cls.min_length or 0
    max_size = cls.max_length
    if cls.regex is None and (not cls.strip_whitespace):
        return st.text(min_size=min_size, max_size=max_size)
    if cls.regex is not None:
        strategy = st.from_regex(cls.regex)
        if cls.strip_whitespace:
            strategy = strategy.filter(lambda s: s == s.strip())
    elif cls.strip_whitespace:
        repeats = '{{{},{}}}'.format(min_size - 2 if min_size > 2 else 0, max_size - 2 if (max_size or 0) > 2 else '')
        if min_size >= 2:
            strategy = st.from_regex(f'\\W.{repeats}\\W')
        elif min_size == 1:
            strategy = st.from_regex(f'\\W(.{repeats}\\W)?')
        else:
            assert min_size == 0
            strategy = st.from_regex(f'(\\W(.{repeats}\\W)?)?')
    if min_size == 0 and max_size is None:
        return strategy
    elif max_size is None:
        return strategy.filter(lambda s: min_size <= len(s))
    return strategy.filter(lambda s: min_size <= len(s) <= max_size)