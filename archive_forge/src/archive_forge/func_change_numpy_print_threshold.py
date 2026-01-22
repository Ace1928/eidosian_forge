import warnings
import numpy
import pytest
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
@pytest.fixture
def change_numpy_print_threshold():
    prev_threshold = numpy.get_printoptions()['threshold']
    numpy.set_printoptions(threshold=50)
    yield prev_threshold
    numpy.set_printoptions(threshold=prev_threshold)