import unittest
from unittest.case import TestCase
import warnings
import numpy as np
from numba import objmode
from numba.core import ir, compiler
from numba.core import errors
from numba.core.compiler import (
from numba.core.compiler_machinery import (
from numba.core.untyped_passes import (
from numba import njit
def test_undefinedtype(self):
    a = ir.UndefinedType()
    b = ir.UndefinedType()
    self.check(a, same=[b])