import math
import re
import numpy as np
import pytest
import scipy.ndimage as ndi
from numpy.testing import (
from skimage import data, draw, transform
from skimage._shared import testing
from skimage.measure._regionprops import (
from skimage.segmentation import slic
def test_regionprops_table_no_regions():
    out = regionprops_table(np.zeros((2, 2), dtype=int), properties=('label', 'area', 'bbox'), separator='+')
    assert len(out) == 6
    assert len(out['label']) == 0
    assert len(out['area']) == 0
    assert len(out['bbox+0']) == 0
    assert len(out['bbox+1']) == 0
    assert len(out['bbox+2']) == 0
    assert len(out['bbox+3']) == 0