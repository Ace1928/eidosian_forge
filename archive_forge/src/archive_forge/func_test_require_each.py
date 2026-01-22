import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
def test_require_each(self):
    id = ['f8', 'i4']
    fd = [None, 'f8', 'c16']
    for idtype, fdtype, flag in itertools.product(id, fd, self.flag_names):
        a = self.generate_all_false(idtype)
        self.set_and_check_flag(flag, fdtype, a)