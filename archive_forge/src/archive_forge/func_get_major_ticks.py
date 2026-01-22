import inspect
import numpy as np
import matplotlib as mpl
from matplotlib import (
from . import art3d, proj3d
def get_major_ticks(self, numticks=None):
    ticks = super().get_major_ticks(numticks)
    for t in ticks:
        for obj in [t.tick1line, t.tick2line, t.gridline, t.label1, t.label2]:
            obj.set_transform(self.axes.transData)
    return ticks