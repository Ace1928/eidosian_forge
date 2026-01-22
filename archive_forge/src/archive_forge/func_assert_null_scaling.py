import itertools
import unittest
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ..optpkg import optional_package
from ..casting import sctypes_aliases, shared_range, type_info
from ..spatialimages import HeaderDataError
from ..spm99analyze import HeaderTypeError, Spm99AnalyzeHeader, Spm99AnalyzeImage
from ..testing import (
from ..volumeutils import _dt_min_max, apply_read_scaling
from . import test_analyze
def assert_null_scaling(self, arr, slope, inter):
    img_class = self.image_class
    input_hdr = img_class.header_class()
    self._set_raw_scaling(input_hdr, slope, inter)
    img = img_class(arr, np.eye(4), input_hdr)
    img_hdr = img.header
    self._set_raw_scaling(input_hdr, slope, inter)
    assert_array_equal(img.get_fdata(), arr)
    fm = bytesio_filemap(img)
    img_fobj = fm['image'].fileobj
    hdr_fobj = img_fobj if not 'header' in fm else fm['header'].fileobj
    img_hdr.write_to(hdr_fobj)
    img_hdr.data_to_fileobj(arr, img_fobj, rescale=False)
    raw_rt_img = img_class.from_file_map(fm)
    assert_array_equal(raw_rt_img.get_fdata(), arr)
    fm = bytesio_filemap(img)
    img.to_file_map(fm)
    rt_img = img_class.from_file_map(fm)
    assert_array_equal(rt_img.get_fdata(), arr)