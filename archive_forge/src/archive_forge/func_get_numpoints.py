from itertools import cycle
import numpy as np
from matplotlib import cbook
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll
def get_numpoints(self, legend):
    if self._numpoints is None:
        return legend.scatterpoints
    else:
        return self._numpoints