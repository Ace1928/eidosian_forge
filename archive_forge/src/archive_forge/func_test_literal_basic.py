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
def test_literal_basic(self):
    self.check_literal_basic([123, 321])
    self.check_literal_basic(['abc', 'cb123'])