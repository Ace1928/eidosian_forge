import logging
from io import BytesIO, StringIO
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from .. import imageglobals
from ..batteryrunners import Report
from ..casting import sctypes
from ..spatialimages import HeaderDataError
from ..volumeutils import Recoder, native_code, swapped_code
from ..wrapstruct import LabeledWrapStruct, WrapStruct, WrapStructError
def test_binblock_is_file(self):
    hdr = self.header_class()
    str_io = BytesIO()
    hdr.write_to(str_io)
    assert str_io.getvalue() == hdr.binaryblock