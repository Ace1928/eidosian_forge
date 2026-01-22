import copy
import operator
import sys
import unittest
import warnings
from collections import defaultdict
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ...testing import assert_arrays_equal, clear_and_catch_warnings
from .. import tractogram as module_tractogram
from ..tractogram import (
def test_setting_affine_to_rasmm(self):
    tractogram = DATA['tractogram'].copy()
    affine = np.diag(range(4))
    tractogram.affine_to_rasmm = None
    assert tractogram.affine_to_rasmm is None
    tractogram.affine_to_rasmm = affine
    assert tractogram.affine_to_rasmm is not affine
    tractogram.affine_to_rasmm = affine.tolist()
    assert_array_equal(tractogram.affine_to_rasmm, affine)
    with pytest.raises(ValueError):
        tractogram.affine_to_rasmm = affine[::2]