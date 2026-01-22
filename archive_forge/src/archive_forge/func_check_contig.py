import sys
import numpy as np
from numpy.core._rational_tests import rational
import pytest
from numpy.testing import (
def check_contig(a, ccontig, fcontig):
    assert_(a.flags.c_contiguous == ccontig)
    assert_(a.flags.f_contiguous == fcontig)