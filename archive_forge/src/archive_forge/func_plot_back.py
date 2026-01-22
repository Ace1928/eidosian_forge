import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Patch3D
from .utils import matplotlib_close_if_inline
def plot_back(self):
    """back half of sphere"""
    u_angle = np.linspace(0, np.pi, 25)
    v_angle = np.linspace(0, np.pi, 25)
    x_dir = np.outer(np.cos(u_angle), np.sin(v_angle))
    y_dir = np.outer(np.sin(u_angle), np.sin(v_angle))
    z_dir = np.outer(np.ones(u_angle.shape[0]), np.cos(v_angle))
    self.axes.plot_surface(x_dir, y_dir, z_dir, rstride=2, cstride=2, color=self.sphere_color, linewidth=0, alpha=self.sphere_alpha)
    self.axes.plot_wireframe(x_dir, y_dir, z_dir, rstride=5, cstride=5, color=self.frame_color, alpha=self.frame_alpha)
    self.axes.plot(1.0 * np.cos(u_angle), 1.0 * np.sin(u_angle), zs=0, zdir='z', lw=self.frame_width, color=self.frame_color)
    self.axes.plot(1.0 * np.cos(u_angle), 1.0 * np.sin(u_angle), zs=0, zdir='x', lw=self.frame_width, color=self.frame_color)