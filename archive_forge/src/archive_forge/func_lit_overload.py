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
@overload_method(DummyType, 'lit')
def lit_overload(self, a):

    def impl(self, a):
        return literally(a)
    return impl