from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nibabel.optpkg import optional_package
from .. import dicomreaders as didr
from .test_dicomwrappers import DATA, EXPECTED_AFFINE, EXPECTED_PARAMS, IO_DATA_PATH
Testing reading DICOM files
