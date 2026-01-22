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
def test_save_from_generator(self):
    tractogram = Tractogram(DATA['streamlines'], affine_to_rasmm=np.eye(4))
    for ext, _ in FORMATS.items():
        filtered = (s for s in tractogram.streamlines if True)
        lazy_tractogram = LazyTractogram(lambda: filtered, affine_to_rasmm=np.eye(4))
        with InTemporaryDirectory():
            filename = 'streamlines' + ext
            nib.streamlines.save(lazy_tractogram, filename)
            tfile = nib.streamlines.load(filename, lazy_load=False)
            assert_tractogram_equal(tfile.tractogram, tractogram)