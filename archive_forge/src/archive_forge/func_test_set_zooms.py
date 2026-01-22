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
def test_set_zooms():
    mgz = load(MGZ_FNAME)
    h = mgz.header
    assert_array_almost_equal(h.get_zooms(), [1, 1, 1, 2])
    h.set_zooms([1, 1, 1, 3])
    assert_array_almost_equal(h.get_zooms(), [1, 1, 1, 3])
    for zooms in ((-1, 1, 1, 1), (1, -1, 1, 1), (1, 1, -1, 1), (1, 1, 1, -1), (1, 1, 1, 1, 5)):
        with pytest.raises(HeaderDataError):
            h.set_zooms(zooms)
    h.set_zooms((1, 1, 1, 0))