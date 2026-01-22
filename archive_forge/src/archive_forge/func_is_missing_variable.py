from typing import Any, Dict, List, Optional, cast
from ..language import (
from ..pyutils import inspect, Undefined
from ..type import (
def is_missing_variable(value_node: ValueNode, variables: Optional[Dict[str, Any]]=None) -> bool:
    """Check if ``value_node`` is a variable not defined in the ``variables`` dict."""
    return isinstance(value_node, VariableNode) and (not variables or variables.get(value_node.name.value, Undefined) is Undefined)