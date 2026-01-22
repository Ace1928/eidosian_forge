from collections import namedtuple
import inspect
import re
import numpy as np
import math
from textwrap import dedent
import unittest
import warnings
from numba.tests.support import (TestCase, override_config,
from numba import jit, njit
from numba.core import types
from numba.core.datamodel import default_manager
from numba.core.errors import NumbaDebugInfoWarning
import llvmlite.binding as llvm
def get_func_attrs(fn):
    cres = fn.overloads[fn.signatures[0]]
    lib = cres.library
    fn = lib._final_module.get_function(cres.fndesc.mangled_name)
    attrs = set(b' '.join(fn.attributes).split())
    return attrs