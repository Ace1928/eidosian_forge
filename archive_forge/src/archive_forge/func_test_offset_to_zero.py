import itertools
import logging
import os
import pickle
import re
from io import BytesIO, StringIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import imageglobals
from ..analyze import AnalyzeHeader, AnalyzeImage
from ..arraywriters import WriterError
from ..casting import sctypes_aliases
from ..nifti1 import Nifti1Header
from ..optpkg import optional_package
from ..spatialimages import HeaderDataError, HeaderTypeError, supported_np_types
from ..testing import (
from ..tmpdirs import InTemporaryDirectory
from . import test_spatialimages as tsi
from . import test_wrapstruct as tws
def test_offset_to_zero(self):
    img_klass = self.image_class
    arr = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
    aff = np.eye(4)
    img = img_klass(arr, aff)
    assert img.header.get_data_offset() == 0
    bytes_map = bytesio_filemap(img_klass)
    img.to_file_map(bytes_map)
    assert img.header.get_data_offset() == 0
    big_off = 1024
    img.header.set_data_offset(big_off)
    assert img.header.get_data_offset() == big_off
    img_rt = bytesio_round_trip(img)
    assert img_rt.dataobj.offset == big_off
    assert img_rt.header.get_data_offset() == 0
    img.header.set_data_offset(big_off)
    img_again = img_klass(arr, aff, img.header)
    assert img_again.header.get_data_offset() == 0