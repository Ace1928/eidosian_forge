import itertools
import sys
import warnings
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nibabel.tmpdirs import InTemporaryDirectory
from ... import load
from ...fileholders import FileHolder
from ...nifti1 import data_type_codes
from ...testing import get_test_data
from .. import (
from .test_parse_gifti_fast import (
def test_gifti_label_rgba():
    rgba = rng.random(4)
    kwargs = dict(zip(['red', 'green', 'blue', 'alpha'], rgba))
    gl1 = GiftiLabel(**kwargs)
    assert_array_equal(rgba, gl1.rgba)
    gl1.red = 2 * gl1.red
    assert not np.allclose(rgba, gl1.rgba)
    gl2 = GiftiLabel()
    gl2.rgba = rgba
    assert_array_equal(rgba, gl2.rgba)
    gl2.blue = 2 * gl2.blue
    assert not np.allclose(rgba, gl2.rgba)

    def assign_rgba(gl, val):
        gl.rgba = val
    gl3 = GiftiLabel(**kwargs)
    pytest.raises(ValueError, assign_rgba, gl3, rgba[:2])
    pytest.raises(ValueError, assign_rgba, gl3, rgba.tolist() + rgba.tolist())
    gl4 = GiftiLabel()
    assert len(gl4.rgba) == 4
    assert np.all([elem is None for elem in gl4.rgba])