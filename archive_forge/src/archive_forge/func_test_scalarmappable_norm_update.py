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
def test_scalarmappable_norm_update():
    norm = mcolors.Normalize()
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap='plasma')
    sm.stale = False
    norm.vmin = 5
    assert sm.stale
    sm.stale = False
    norm.vmax = 5
    assert sm.stale
    sm.stale = False
    norm.clip = True
    assert sm.stale
    norm = mcolors.CenteredNorm()
    sm.norm = norm
    sm.stale = False
    norm.vcenter = 1
    assert sm.stale
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
    sm.set_norm(norm)
    sm.stale = False
    norm.vcenter = 1
    assert sm.stale