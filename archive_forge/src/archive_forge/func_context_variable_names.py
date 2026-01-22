from contextvars import Context, ContextVar, copy_context
from typing import Any, Dict, List
@classmethod
def context_variable_names(cls) -> List[str]:
    """Returns a list of variable names set for this call context.

        Returns
        -------
        names: List[str]
            A list of variable names set for this call context.
        """
    name_value_map = CallContext._get_map()
    return list(name_value_map.keys())