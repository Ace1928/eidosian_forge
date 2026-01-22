import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_signed_overflow_bounds(self):
    self.do_signed_overflow_bounds(np.byte)
    self.do_signed_overflow_bounds(np.short)
    self.do_signed_overflow_bounds(np.intc)
    self.do_signed_overflow_bounds(np.int_)
    self.do_signed_overflow_bounds(np.longlong)