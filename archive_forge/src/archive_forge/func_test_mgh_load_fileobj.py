import io
import os
import pathlib
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from ... import imageglobals
from ...fileholders import FileHolder
from ...openers import ImageOpener
from ...spatialimages import HeaderDataError
from ...testing import data_path
from ...tests import test_spatialimages as tsi
from ...tests import test_wrapstruct as tws
from ...tmpdirs import InTemporaryDirectory
from ...volumeutils import sys_is_le
from ...wrapstruct import WrapStructError
from .. import load, save
from ..mghformat import MGHError, MGHHeader, MGHImage
def test_mgh_load_fileobj():
    img = MGHImage.load(MGZ_FNAME)
    assert pathlib.Path(img.dataobj.file_like) == pathlib.Path(MGZ_FNAME)
    with ImageOpener(MGZ_FNAME) as fobj:
        contents = fobj.read()
    bio = io.BytesIO(contents)
    fm = MGHImage.make_file_map(mapping=dict(image=bio))
    img2 = MGHImage.from_file_map(fm)
    assert img2.dataobj.file_like is bio
    assert_array_equal(img.get_fdata(), img2.get_fdata())