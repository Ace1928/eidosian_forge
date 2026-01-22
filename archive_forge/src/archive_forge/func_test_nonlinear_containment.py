import re
import numpy as np
from numpy.testing import assert_array_equal
import pytest
from matplotlib import patches
from matplotlib.path import Path
from matplotlib.patches import Polygon
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.backend_bases import MouseEvent
def test_nonlinear_containment():
    fig, ax = plt.subplots()
    ax.set(xscale='log', ylim=(0, 1))
    polygon = ax.axvspan(1, 10)
    assert polygon.get_path().contains_point(ax.transData.transform((5, 0.5)), ax.transData)
    assert not polygon.get_path().contains_point(ax.transData.transform((0.5, 0.5)), ax.transData)
    assert not polygon.get_path().contains_point(ax.transData.transform((50, 0.5)), ax.transData)