import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
def test_white_tophat04(self):
    array = numpy.eye(5, dtype=numpy.bool_)
    structure = numpy.ones((3, 3), dtype=numpy.bool_)
    output = numpy.empty_like(array, dtype=numpy.float64)
    ndimage.white_tophat(array, structure=structure, output=output)