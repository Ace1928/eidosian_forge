from math import nan
from typing import Any, Callable, Dict, Optional, Union
from ..language import (
from ..pyutils import inspect, Undefined
def value_from_variable(value_node: VariableNode, variables: Optional[Dict[str, Any]]) -> Any:
    variable_name = value_node.name.value
    if not variables:
        return Undefined
    return variables.get(variable_name, Undefined)