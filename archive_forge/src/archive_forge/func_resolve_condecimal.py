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
@resolves(pydantic.ConstrainedDecimal)
def resolve_condecimal(cls):
    min_value = cls.ge
    max_value = cls.le
    if cls.gt is not None:
        assert min_value is None, 'Set `gt` or `ge`, but not both'
        min_value = cls.gt
    if cls.lt is not None:
        assert max_value is None, 'Set `lt` or `le`, but not both'
        max_value = cls.lt
    s = st.decimals(min_value, max_value, allow_nan=False, places=cls.decimal_places)
    if cls.lt is not None:
        s = s.filter(lambda d: d < cls.lt)
    if cls.gt is not None:
        s = s.filter(lambda d: cls.gt < d)
    return s