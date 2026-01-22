from tempfile import TemporaryFile
import numpy as np
from packaging.version import parse as parse_version
import pytest
import matplotlib as mpl
from matplotlib import dviread
from matplotlib.testing import _has_tex_package
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
import matplotlib.pyplot as plt
def test_mathdefault():
    plt.rcParams['axes.formatter.use_mathtext'] = True
    fig = plt.figure()
    fig.add_subplot().set_xlim(-1, 1)
    mpl.rcParams['text.usetex'] = True
    fig.canvas.draw()