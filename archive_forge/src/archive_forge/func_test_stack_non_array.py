import os
import pathlib
import tempfile
import numpy as np
import pytest
from skimage import io
from skimage._shared.testing import assert_array_equal, fetch
from skimage.data import data_dir
def test_stack_non_array():
    with pytest.raises(ValueError):
        io.push([[1, 2, 3]])