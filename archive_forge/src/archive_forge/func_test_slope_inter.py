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
def test_slope_inter(self):
    hdr = self.header_class()
    assert hdr.get_slope_inter() == (None, None)
    for slinter in ((None,), (None, None), (np.nan, np.nan), (np.nan, None), (None, np.nan), (1.0,), (1.0, None), (None, 0), (1.0, 0)):
        hdr.set_slope_inter(*slinter)
        assert hdr.get_slope_inter() == (None, None)
    with pytest.raises(HeaderTypeError):
        hdr.set_slope_inter(1.1)
    with pytest.raises(HeaderTypeError):
        hdr.set_slope_inter(1.0, 0.1)