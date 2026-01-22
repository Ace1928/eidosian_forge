import collections
import math
import numbers
from typing import Any, Dict as PythonDict, Hashable, List as PythonList, Optional, Sequence, Tuple as PythonTuple, Type
import weakref
from tensorflow.core.function.trace_type import default_types_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace
def cast_and_return_whether_casted(trace_types, values, context) -> PythonTuple[PythonList[Any], bool]:
    did_cast = False
    casted_values = []
    for t, v in zip(trace_types, values):
        casted_v = t._cast(v, context)
        casted_values.append(casted_v)
        if casted_v is not v:
            did_cast = True
    return (casted_values, did_cast)