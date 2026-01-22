import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_object_array_of_0d(self):
    assert_raises(ValueError, histogram, [np.array(0.4) for i in range(10)] + [-np.inf])
    assert_raises(ValueError, histogram, [np.array(0.4) for i in range(10)] + [np.inf])
    np.histogram([np.array(0.5) for i in range(10)] + [0.500000000000001])
    np.histogram([np.array(0.5) for i in range(10)] + [0.5])