import numpy as np
import pytest
from pandas.core.indexers import (
def test_validate_indices_ok(self):
    indices = np.asarray([0, 1])
    validate_indices(indices, 2)
    validate_indices(indices[:0], 0)
    validate_indices(np.array([-1, -1]), 0)