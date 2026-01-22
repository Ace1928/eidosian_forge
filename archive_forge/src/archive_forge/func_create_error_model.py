from collections import namedtuple
from collections.abc import Iterable
import itertools
import hashlib
from llvmlite import ir
from numba.core import types, cgutils, errors
from numba.core.base import PYOBJECT, GENERIC_POINTER
def create_error_model(model_name, context):
    """
    Create an error model instance for the given target context.
    """
    return error_models[model_name](context.call_conv)