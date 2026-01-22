import inspect
import math
import operator
import sys
import pickle
import multiprocessing
import ctypes
import warnings
import re
import numpy as np
from llvmlite import ir
import numba
from numba import njit, jit, vectorize, guvectorize, objmode
from numba.core import types, errors, typing, compiler, cgutils
from numba.core.typed_passes import type_inference_stage
from numba.core.registry import cpu_target
from numba.core.imputils import lower_constant
from numba.tests.support import (
from numba.core.errors import LoweringError
import unittest
from numba.extending import (
from numba.core.typing.templates import (
from .pdlike_usecase import Index, Series
@lower_builtin('print_item', MyDummyType)
def print_dummy(context, builder, sig, args):
    [x] = args
    pyapi = context.get_python_api(builder)
    strobj = pyapi.unserialize(pyapi.serialize_object('hello!'))
    pyapi.print_object(strobj)
    pyapi.decref(strobj)
    return context.get_dummy_value()