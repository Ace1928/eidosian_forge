import collections
from typing import Any, NamedTuple, Optional
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.function.polymorphism import type_dispatch
def generalize(self, context: FunctionContext, function_type: function_type_lib.FunctionType) -> function_type_lib.FunctionType:
    """Try to generalize a FunctionType within a FunctionContext."""
    if context in self._dispatch_dict:
        return self._dispatch_dict[context].try_generalizing_function_type(function_type)
    else:
        return function_type