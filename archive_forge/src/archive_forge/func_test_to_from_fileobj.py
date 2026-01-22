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
def test_to_from_fileobj(self):
    hdr = self.header_class()
    str_io = io.BytesIO()
    hdr.write_to(str_io)
    str_io.seek(0)
    hdr2 = self.header_class.from_fileobj(str_io)
    assert hdr2.endianness == '>'
    assert hdr2.binaryblock == hdr.binaryblock