import os
import sys
import pytest
import warnings
import weakref
import numpy as np
import pyarrow as pa
def test_tensor_hashing():
    with pytest.raises(TypeError, match='unhashable'):
        hash(pa.Tensor.from_numpy(np.arange(10)))