import collections
from typing import Optional, Iterable
from tensorflow.core.function.polymorphism import function_type
Returns a generalized subtype of the one given.

    This heuristic aims to reduce the number of future traces by computing a
    type that represents more general function inputs.

    The original "experimental_relax_shapes" heuristic identified a known type
    which shared a common subtype with the current unknown type and then
    traced with that common subtype. However, the notion of "common subtype"
    was only limited to shapes. This heuristic extends that to FunctionType.

    Returns `target` if a generalized subtype can not be found.

    Args:
      target: The FunctionType to generalize
    