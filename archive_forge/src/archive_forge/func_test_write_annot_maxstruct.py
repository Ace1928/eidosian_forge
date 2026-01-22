import getpass
import hashlib
import os
import struct
import time
import unittest
from os.path import isdir
from os.path import join as pjoin
from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_allclose
from ...fileslice import strided_scalar
from ...testing import clear_and_catch_warnings
from ...tests.nibabel_data import get_nibabel_data, needs_nibabel_data
from ...tmpdirs import InTemporaryDirectory
from .. import (
from ..io import _pack_rgb
def test_write_annot_maxstruct():
    """Test writing ANNOT files with repeated labels"""
    with InTemporaryDirectory():
        nlabels = 3
        names = [f'label {l}' for l in range(1, nlabels + 1)]
        labels = np.array([1, 1, 1], dtype=np.int32)
        rgba = np.array(np.random.randint(0, 255, (nlabels, 4)), dtype=np.int32)
        annot_path = 'c.annot'
        write_annot(annot_path, labels, rgba, names)
        rt_labels, rt_ctab, rt_names = read_annot(annot_path)
        assert np.array_equal(labels, rt_labels)
        assert np.array_equal(rgba, rt_ctab[:, :4])
        assert names == [n.decode('ascii') for n in rt_names]