import itertools
import sys
import warnings
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nibabel.tmpdirs import InTemporaryDirectory
from ... import load
from ...fileholders import FileHolder
from ...nifti1 import data_type_codes
from ...testing import get_test_data
from .. import (
from .test_parse_gifti_fast import (
def test_gifti_coord(capsys):
    from ..gifti import GiftiCoordSystem
    gcs = GiftiCoordSystem()
    assert gcs.xform is not None
    gcs.xform = None
    gcs.print_summary()
    captured = capsys.readouterr()
    assert captured.out == '\n'.join(['Dataspace:  NIFTI_XFORM_UNKNOWN', 'XFormSpace:  NIFTI_XFORM_UNKNOWN', 'Affine Transformation Matrix: ', ' None\n'])
    gcs.to_xml()