from ...testing import utils
from ..confounds import TSNR
from .. import misc
import pytest
import numpy.testing as npt
from unittest import mock
import nibabel as nb
import numpy as np
import os
def assert_unchanged(self, expected_ranges):
    for key, (min_, max_) in expected_ranges.items():
        data = np.asarray(nb.load(self.out_filenames[key]).dataobj)
        npt.assert_almost_equal(np.amin(data), min_, decimal=1)
        npt.assert_almost_equal(np.amax(data), max_, decimal=1)