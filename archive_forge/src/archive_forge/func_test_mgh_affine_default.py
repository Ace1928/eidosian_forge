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
def test_mgh_affine_default():
    hdr = MGHHeader()
    hdr['goodRASFlag'] = 0
    hdr2 = MGHHeader(hdr.binaryblock)
    assert hdr2['goodRASFlag'] == 1
    assert_array_equal(hdr['Mdc'], hdr2['Mdc'])
    assert_array_equal(hdr['Pxyz_c'], hdr2['Pxyz_c'])