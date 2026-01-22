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
def test_colormap_equals():
    cmap = mpl.colormaps['plasma']
    cm_copy = cmap.copy()
    assert cm_copy is not cmap
    assert cm_copy == cmap
    cm_copy.set_bad('y')
    assert cm_copy != cmap
    cm_copy._lut = cm_copy._lut[:10, :]
    assert cm_copy != cmap
    cm_copy = cmap.copy()
    cm_copy.name = 'Test'
    assert cm_copy == cmap
    cm_copy = cmap.copy()
    cm_copy.colorbar_extend = not cmap.colorbar_extend
    assert cm_copy != cmap