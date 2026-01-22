from ipywidgets import Layout
from traitlets import List, Enum, Int, Bool
from traittypes import DataFrame
from bqplot import Figure, LinearScale, Lines, Label
from bqplot.marks import CATEGORY10
import numpy as np
def update_fill(self, *args):
    if self.fill:
        with self.loops.hold_sync():
            self.loops.fill = 'inside'
            self.loops.fill_opacities = [0.2] * len(self.loops.y)
    else:
        self.loops.fill = 'none'
        self.loops.fill_opacities = [0.0] * len(self.loops.y)