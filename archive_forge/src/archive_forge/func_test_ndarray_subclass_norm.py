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
def test_ndarray_subclass_norm():

    class MyArray(np.ndarray):

        def __isub__(self, other):
            raise RuntimeError

        def __add__(self, other):
            raise RuntimeError
    data = np.arange(-10, 10, 1, dtype=float).reshape((10, 2))
    mydata = data.view(MyArray)
    for norm in [mcolors.Normalize(), mcolors.LogNorm(), mcolors.SymLogNorm(3, vmax=5, linscale=1, base=np.e), mcolors.Normalize(vmin=mydata.min(), vmax=mydata.max()), mcolors.SymLogNorm(3, vmin=mydata.min(), vmax=mydata.max(), base=np.e), mcolors.PowerNorm(1)]:
        assert_array_equal(norm(mydata), norm(data))
        fig, ax = plt.subplots()
        ax.imshow(mydata, norm=norm)
        fig.canvas.draw()