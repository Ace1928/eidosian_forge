import collections
from typing import Optional, Iterable
from tensorflow.core.function.polymorphism import function_type
def try_generalizing_function_type(self, target: function_type.FunctionType) -> function_type.FunctionType:
    """Returns a generalized subtype of the one given.

    This heuristic aims to reduce the number of future traces by computing a
    type that represents more general function inputs.

    The original "experimental_relax_shapes" heuristic identified a known type
    which shared a common subtype with the current unknown type and then
    traced with that common subtype. However, the notion of "common subtype"
    was only limited to shapes. This heuristic extends that to FunctionType.

    Returns `target` if a generalized subtype can not be found.

    Args:
      target: The FunctionType to generalize
    """
    relaxed = target
    for other in self._dispatch_table:
        subtype = relaxed.most_specific_common_subtype([other])
        if subtype is not None:
            relaxed = subtype
    return relaxed