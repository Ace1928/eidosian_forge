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
def test_usetex_with_underscore():
    plt.rcParams['text.usetex'] = True
    df = {'a_b': range(5)[::-1], 'c': range(5)}
    fig, ax = plt.subplots()
    ax.plot('c', 'a_b', data=df)
    ax.legend()
    ax.text(0, 0, 'foo_bar', usetex=True)
    plt.draw()