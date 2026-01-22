import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Patch3D
from .utils import matplotlib_close_if_inline
def plot_axes(self):
    """axes"""
    span = np.linspace(-1.0, 1.0, 2)
    self.axes.plot(span, 0 * span, zs=0, zdir='z', label='X', lw=self.frame_width, color=self.frame_color)
    self.axes.plot(0 * span, span, zs=0, zdir='z', label='Y', lw=self.frame_width, color=self.frame_color)
    self.axes.plot(0 * span, span, zs=0, zdir='y', label='Z', lw=self.frame_width, color=self.frame_color)