import pytest
import numpy as np
from numpy.core import (
from numpy.core.shape_base import (_block_dispatcher, _block_setup,
from numpy.testing import (
def test_block_total_size_estimate(self, block):
    _, _, _, total_size = _block_setup([1])
    assert total_size == 1
    _, _, _, total_size = _block_setup([[1]])
    assert total_size == 1
    _, _, _, total_size = _block_setup([[1, 1]])
    assert total_size == 2
    _, _, _, total_size = _block_setup([[1], [1]])
    assert total_size == 2
    _, _, _, total_size = _block_setup([[1, 2], [3, 4]])
    assert total_size == 4