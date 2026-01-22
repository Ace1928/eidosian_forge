import datetime
from io import BytesIO
import os
import shutil
import numpy as np
from packaging.version import parse as parse_version
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.testing import _has_tex_package, _check_for_pgf
from matplotlib.testing.exceptions import ImageComparisonFailure
from matplotlib.testing.compare import compare_images
from matplotlib.backends.backend_pgf import PdfPages
from matplotlib.testing.decorators import (
from matplotlib.testing._markers import (
@needs_pgf_xelatex
@mpl.style.context('default')
@pytest.mark.backend('pgf')
def test_pathclip():
    np.random.seed(19680801)
    mpl.rcParams.update({'font.family': 'serif', 'pgf.rcfonts': False})
    fig, axs = plt.subplots(1, 2)
    axs[0].plot([0.0, 1e+100], [0.0, 1e+100])
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(0, 1)
    axs[1].scatter([0, 1], [1, 1])
    axs[1].hist(np.random.normal(size=1000), bins=20, range=[-10, 10])
    axs[1].set_xscale('log')
    fig.savefig(BytesIO(), format='pdf')