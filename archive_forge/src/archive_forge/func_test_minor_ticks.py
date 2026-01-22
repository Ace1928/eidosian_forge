import functools
import itertools
import platform
import pytest
from mpl_toolkits.mplot3d import Axes3D, axes3d, proj3d, art3d
import matplotlib as mpl
from matplotlib.backend_bases import (MouseButton, MouseEvent,
from matplotlib import cm
from matplotlib import colors as mcolors, patches as mpatch
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.testing.widgets import mock_event
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np
@image_comparison(['minor_ticks.png'], style='mpl20')
def test_minor_ticks():
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xticks([0.25], minor=True)
    ax.set_xticklabels(['quarter'], minor=True)
    ax.set_yticks([0.33], minor=True)
    ax.set_yticklabels(['third'], minor=True)
    ax.set_zticks([0.5], minor=True)
    ax.set_zticklabels(['half'], minor=True)