import numpy as np
import numba
import unittest
from numba.tests.support import TestCase
from numba import njit
from numba.core import types, errors, cgutils
from numba.core.typing import signature
from numba.core.datamodel import models
from numba.core.extending import (
from numba.misc.special import literally
@unittest.expectedFailure
def test_literally_defaults_inner(self):

    @njit
    def foo(a, b=1):
        return (a, literally(b))

    @njit
    def bar(a):
        return foo(a) + 1
    bar(1)