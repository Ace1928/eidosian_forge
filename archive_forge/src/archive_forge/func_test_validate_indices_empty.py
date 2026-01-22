import numpy as np
import pytest
from pandas.core.indexers import (
def test_validate_indices_empty(self):
    with pytest.raises(IndexError, match='indices are out'):
        validate_indices(np.array([0, 1]), 0)