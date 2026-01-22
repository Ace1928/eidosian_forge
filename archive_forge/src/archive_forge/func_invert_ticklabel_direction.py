from operator import methodcaller
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.text as mtext
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import (
from .axisline_style import AxislineStyle
def invert_ticklabel_direction(self):
    self._ticklabel_add_angle = (self._ticklabel_add_angle + 180) % 360
    self.major_ticklabels.invert_axis_direction()
    self.minor_ticklabels.invert_axis_direction()