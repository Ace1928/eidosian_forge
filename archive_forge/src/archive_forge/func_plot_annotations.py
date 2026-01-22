import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Patch3D
from .utils import matplotlib_close_if_inline
def plot_annotations(self):
    """Plot annotations"""
    for annotation in self.annotations:
        vec = annotation['position']
        opts = {'fontsize': self.font_size, 'color': self.font_color, 'horizontalalignment': 'center', 'verticalalignment': 'center'}
        opts.update(annotation['opts'])
        self.axes.text(vec[1], -vec[0], vec[2], annotation['text'], **opts)