import math
import numbers
from typing import cast, Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Union
import numpy as np
import sympy
from cirq_google.api import v2
from cirq_google.ops import InternalGate
def float_arg_to_proto(value: ARG_LIKE, *, arg_function_language: Optional[str]=None, out: Optional[v2.program_pb2.FloatArg]=None) -> v2.program_pb2.FloatArg:
    """Writes an argument value into an FloatArg proto.

    Note that the FloatArg proto is a slimmed down form of the
    Arg proto, so this proto should only be used when the argument
    is known to be a float or expression that resolves to a float.

    Args:
        value: The value to encode.  This must be a float or compatible
            sympy expression. Strings and repeated booleans are not allowed.
        arg_function_language: The language to use when encoding functions. If
            this is set to None, it will be set to the minimal language
            necessary to support the features that were actually used.
        out: The proto to write the result into. Defaults to a new instance.

    Returns:
        The proto that was written into.
    """
    msg = v2.program_pb2.FloatArg() if out is None else out
    if isinstance(value, FLOAT_TYPES):
        msg.float_value = float(value)
    else:
        _arg_func_to_proto(value, arg_function_language, msg)
    return msg