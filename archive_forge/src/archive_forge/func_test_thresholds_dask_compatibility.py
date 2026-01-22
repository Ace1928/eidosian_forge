import math
import numpy as np
import pytest
from numpy.testing import (
from scipy import ndimage as ndi
from skimage import data, util
from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.draw import disk
from skimage.exposure import histogram
from skimage.filters._multiotsu import (
from skimage.filters.thresholding import (
@pytest.mark.parametrize('thresholding, lower, upper', [(threshold_otsu, 101, 103), (threshold_yen, 145, 147), (threshold_isodata, 101, 103), (threshold_mean, 128, 130), (threshold_triangle, 41, 43), (threshold_minimum, 84, 86)])
def test_thresholds_dask_compatibility(thresholding, lower, upper):
    pytest.importorskip('dask', reason='dask python library is not installed')
    import dask.array as da
    dask_camera = da.from_array(data.camera(), chunks=(256, 256))
    assert lower < float(thresholding(dask_camera)) < upper