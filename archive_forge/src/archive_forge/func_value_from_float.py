from math import nan
from typing import Any, Callable, Dict, Optional, Union
from ..language import (
from ..pyutils import inspect, Undefined
def value_from_float(value_node: FloatValueNode, _variables: Any) -> Any:
    try:
        return float(value_node.value)
    except ValueError:
        return nan