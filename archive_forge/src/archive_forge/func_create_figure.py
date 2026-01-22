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
def create_figure():
    plt.figure()
    x = np.linspace(0, 1, 15)
    plt.plot(x, x ** 2, 'b-')
    plt.plot(x, 1 - x ** 2, 'g>')
    plt.fill_between([0.0, 0.4], [0.4, 0.0], hatch='//', facecolor='lightgray', edgecolor='red')
    plt.fill([3, 3, 0.8, 0.8, 3], [2, -2, -2, 0, 2], 'b')
    plt.plot([0.9], [0.5], 'ro', markersize=3)
    plt.text(0.9, 0.5, 'unicode (ü, °, §) and math ($\\mu_i = x_i^2$)', ha='right', fontsize=20)
    plt.ylabel('sans-serif, blue, $\\frac{\\sqrt{x}}{y^2}$..', family='sans-serif', color='blue')
    plt.text(1, 1, 'should be clipped as default clip_box is Axes bbox', fontsize=20, clip_on=True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)