from __future__ import annotations
import logging # isort:skip
import datetime
from typing import Any, Union
from ...util.serialization import (
from .bases import Init, Property
from .primitive import bokeh_integer_types
from .singletons import Undefined
@staticmethod
def is_timestamp(value: Any) -> bool:
    return isinstance(value, (float, *bokeh_integer_types)) and (not isinstance(value, bool))