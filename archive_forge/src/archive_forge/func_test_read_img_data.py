import pathlib
import shutil
from os.path import dirname
from os.path import join as pjoin
from tempfile import TemporaryDirectory
import numpy as np
from .. import (
from ..filebasedimages import ImageFileError
from ..loadsave import _signature_matches_extension, load, read_img_data
from ..openers import Opener
from ..optpkg import optional_package
from ..testing import deprecated_to, expires
from ..tmpdirs import InTemporaryDirectory
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
@expires('5.0.0')
def test_read_img_data():
    fnames_test = ['example4d.nii.gz', 'example_nifti2.nii.gz', 'minc1_1_scale.mnc', 'minc1_4d.mnc', 'test.mgz', 'tiny.mnc']
    fnames_test += [pathlib.Path(p) for p in fnames_test]
    for fname in fnames_test:
        fpath = pjoin(data_path, fname)
        if isinstance(fname, pathlib.Path):
            fpath = pathlib.Path(fpath)
        img = load(fpath)
        data = img.get_fdata()
        with deprecated_to('5.0.0'):
            data2 = read_img_data(img)
        assert_array_equal(data, data2)
        dao = img.dataobj
        if hasattr(dao, 'slope') and hasattr(img.header, 'raw_data_from_fileobj'):
            assert (dao.slope, dao.inter) == (1, 0)
            with deprecated_to('5.0.0'):
                assert_array_equal(read_img_data(img, prefer='unscaled'), data)
        with TemporaryDirectory() as tmpdir:
            up_fpath = pjoin(tmpdir, str(fname).upper())
            if isinstance(fname, pathlib.Path):
                up_fpath = pathlib.Path(up_fpath)
            shutil.copyfile(fpath, up_fpath)
            img = load(up_fpath)
            assert_array_equal(img.dataobj, data)
            del img