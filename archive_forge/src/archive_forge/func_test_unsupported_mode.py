import pytest
import copy
import numpy as np
from skimage._shared.testing import assert_array_equal
from skimage import data
from skimage.feature import BRIEF, corner_peaks, corner_harris
from skimage._shared import testing
def test_unsupported_mode():
    with testing.raises(ValueError):
        BRIEF(mode='foobar')