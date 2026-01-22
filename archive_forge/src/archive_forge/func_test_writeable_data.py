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
def test_writeable_data(self):
    data = DATA['simple_tractogram']
    for key in ('simple_tck_fname', 'simple_tck_big_endian_fname'):
        for lazy_load in [False, True]:
            tck = TckFile.load(DATA[key], lazy_load=lazy_load)
            for actual, expected_tgi in zip(tck.streamlines, data):
                assert_array_equal(actual, expected_tgi.streamline)
                assert actual.flags.writeable
                actual[0, 0] = 99