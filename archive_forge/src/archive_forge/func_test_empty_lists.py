import pytest
import numpy as np
from numpy.core import (
from numpy.core.shape_base import (_block_dispatcher, _block_setup,
from numpy.testing import (
def test_empty_lists(self, block):
    assert_raises_regex(ValueError, 'empty', block, [])
    assert_raises_regex(ValueError, 'empty', block, [[]])
    assert_raises_regex(ValueError, 'empty', block, [[1], []])