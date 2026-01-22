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
def test_save_tractogram_file(self):
    tractogram = Tractogram(DATA['streamlines'], affine_to_rasmm=np.eye(4))
    trk_file = trk.TrkFile(tractogram)
    with self.assertRaises(ValueError):
        nib.streamlines.save(trk_file, 'dummy.trk', header={})
    with pytest.warns(ExtensionWarning, match='extension'):
        trk_file = trk.TrkFile(tractogram)
        with self.assertRaises(ValueError):
            nib.streamlines.save(trk_file, 'dummy.tck', header={})
    with InTemporaryDirectory():
        nib.streamlines.save(trk_file, 'dummy.trk')
        tfile = nib.streamlines.load('dummy.trk', lazy_load=False)
        assert_tractogram_equal(tfile.tractogram, tractogram)