from itertools import cycle
import numpy as np
from matplotlib import cbook
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll
def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
    p = Rectangle(xy=(-xdescent, -ydescent), width=width, height=height)
    self.update_prop(p, orig_handle, legend)
    p.set_transform(trans)
    return [p]