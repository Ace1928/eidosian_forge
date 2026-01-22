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
@decor
def inner_fac(n, value):
    if n < 1:
        return literally(value)
    return n * outer_fac(n - 1, value)