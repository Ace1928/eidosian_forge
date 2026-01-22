from functools import wraps
import inspect
import operator
from numba.core.extending import overload
from numba.core.types import ClassInstanceType
@class_instance_overload(float)
def class_float(x):
    options = [try_call_method(x, '__float__')]
    if '__index__' in x.jit_methods:
        options.append(lambda x: float(x.__index__()))
    return take_first(*options)