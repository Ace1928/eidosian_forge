import os
import tempfile
import unittest
import warnings
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.compat.py3k import asbytes
import nibabel as nib
from nibabel.testing import clear_and_catch_warnings, data_path, error_warnings
from nibabel.tmpdirs import InTemporaryDirectory
from .. import FORMATS, trk
from ..tractogram import LazyTractogram, Tractogram
from ..tractogram_file import ExtensionWarning, TractogramFile
from .test_tractogram import assert_tractogram_equal
def test_load_empty_file(self):
    for lazy_load in [False, True]:
        for empty_filename in DATA['empty_filenames']:
            tfile = nib.streamlines.load(empty_filename, lazy_load=lazy_load)
            assert isinstance(tfile, TractogramFile)
            if lazy_load:
                assert type(tfile.tractogram), Tractogram
            else:
                assert type(tfile.tractogram), LazyTractogram
            with pytest.warns(Warning) if lazy_load else error_warnings():
                assert_tractogram_equal(tfile.tractogram, DATA['empty_tractogram'])