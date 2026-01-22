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
def test_double_register_builtin_cmap():
    name = 'viridis'
    match = f'Re-registering the builtin cmap {name!r}.'
    with pytest.raises(ValueError, match=match):
        matplotlib.colormaps.register(mpl.colormaps[name], name=name, force=True)
    with pytest.raises(ValueError, match='A colormap named "viridis"'):
        with pytest.warns(mpl.MatplotlibDeprecationWarning):
            cm.register_cmap(name, mpl.colormaps[name])
    if parse_version(pytest.__version__).major < 8:
        with pytest.warns(UserWarning):
            cm.register_cmap(name, mpl.colormaps[name], override_builtin=True)
    else:
        with pytest.warns(UserWarning), pytest.warns(mpl.MatplotlibDeprecationWarning):
            cm.register_cmap(name, mpl.colormaps[name], override_builtin=True)