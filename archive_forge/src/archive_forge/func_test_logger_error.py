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
def test_logger_error(self):
    HC = self.header_class
    hdr = HC()
    str_io = StringIO()
    logger = logging.getLogger('test.logger')
    logger.addHandler(logging.StreamHandler(str_io))
    hdr['datatype'] = 16
    hdr['bitpix'] = 16
    logger.setLevel(10)
    log_cache = (imageglobals.logger, imageglobals.error_level)
    try:
        imageglobals.logger = logger
        hdr.copy().check_fix()
        assert str_io.getvalue() == 'bitpix does not match datatype; setting bitpix to match datatype\n'
        imageglobals.error_level = 10
        with pytest.raises(HeaderDataError):
            hdr.copy().check_fix()
    finally:
        imageglobals.logger, imageglobals.error_level = log_cache