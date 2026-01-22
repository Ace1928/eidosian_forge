import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Patch3D
from .utils import matplotlib_close_if_inline
def plot_vectors(self):
    """Plot vector"""
    for k in range(len(self.vectors)):
        xs3d = self.vectors[k][1] * np.array([0, 1])
        ys3d = -self.vectors[k][0] * np.array([0, 1])
        zs3d = self.vectors[k][2] * np.array([0, 1])
        color = self.vector_color[np.mod(k, len(self.vector_color))]
        if self.vector_style == '':
            self.axes.plot(xs3d, ys3d, zs3d, zs=0, zdir='z', label='Z', lw=self.vector_width, color=color)
        else:
            arr = Arrow3D(xs3d, ys3d, zs3d, mutation_scale=self.vector_mutation, lw=self.vector_width, arrowstyle=self.vector_style, color=color)
            self.axes.add_artist(arr)