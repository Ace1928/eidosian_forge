import copy
import itertools
import unittest.mock
from packaging.version import parse as parse_version
from io import BytesIO
import numpy as np
from PIL import Image
import pytest
import base64
from numpy.testing import assert_array_equal, assert_array_almost_equal
from matplotlib import cbook, cm
import matplotlib
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
from matplotlib.rcsetup import cycler
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_colormaps_get_cmap():
    cr = mpl.colormaps
    assert cr.get_cmap('plasma') == cr['plasma']
    assert cr.get_cmap(cr['magma']) == cr['magma']
    assert cr.get_cmap(None) == cr[mpl.rcParams['image.cmap']]
    bad_cmap = 'AardvarksAreAwkward'
    with pytest.raises(ValueError, match=bad_cmap):
        cr.get_cmap(bad_cmap)
    with pytest.raises(TypeError, match='object'):
        cr.get_cmap(object())