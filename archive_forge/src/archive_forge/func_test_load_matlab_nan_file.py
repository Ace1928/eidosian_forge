import os
import unittest
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ...testing import data_path, error_warnings
from .. import tck as tck_module
from ..array_sequence import ArraySequence
from ..tck import TckFile
from ..tractogram import Tractogram
from ..tractogram_file import DataError, HeaderError, HeaderWarning
from .test_tractogram import assert_tractogram_equal
def test_load_matlab_nan_file(self):
    for lazy_load in [False, True]:
        tck = TckFile.load(DATA['matlab_nan_tck_fname'], lazy_load=lazy_load)
        streamlines = list(tck.tractogram.streamlines)
        assert len(streamlines) == 1
        assert streamlines[0].shape == (108, 3)