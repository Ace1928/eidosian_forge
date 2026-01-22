import copy
import os
import sys
import unittest
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ...testing import assert_arr_dict_equal, clear_and_catch_warnings, data_path, error_warnings
from .. import trk as trk_module
from ..header import Field
from ..tractogram import Tractogram
from ..tractogram_file import HeaderError, HeaderWarning
from ..trk import (
from .test_tractogram import assert_tractogram_equal
def test_load_trk_version_1(self):
    trk_struct, trk_bytes = self.trk_with_bytes()
    trk_struct[Field.VOXEL_TO_RASMM] = np.diag([2, 3, 4, 1])
    trk = TrkFile.load(BytesIO(trk_bytes))
    assert_array_equal(trk.affine, np.diag([2, 3, 4, 1]))
    trk_struct['version'] = 1
    with pytest.warns(HeaderWarning, match='identity'):
        trk = TrkFile.load(BytesIO(trk_bytes))
    assert_array_equal(trk.affine, np.eye(4))
    assert_array_equal(trk.header['version'], 1)