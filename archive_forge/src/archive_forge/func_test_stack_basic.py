import os
import pathlib
import tempfile
import numpy as np
import pytest
from skimage import io
from skimage._shared.testing import assert_array_equal, fetch
from skimage.data import data_dir
def test_stack_basic():
    x = np.arange(12).reshape(3, 4)
    io.push(x)
    assert_array_equal(io.pop(), x)