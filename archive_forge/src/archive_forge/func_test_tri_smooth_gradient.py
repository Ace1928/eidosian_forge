import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
@image_comparison(['tri_smooth_gradient.png'], remove_text=True, tol=0.092)
def test_tri_smooth_gradient():

    def dipole_potential(x, y):
        """An electric dipole potential V."""
        r_sq = x ** 2 + y ** 2
        theta = np.arctan2(y, x)
        z = np.cos(theta) / r_sq
        return (np.max(z) - z) / (np.max(z) - np.min(z))
    n_angles = 30
    n_radii = 10
    min_radius = 0.2
    radii = np.linspace(min_radius, 0.95, n_radii)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    angles[:, 1::2] += np.pi / n_angles
    x = (radii * np.cos(angles)).flatten()
    y = (radii * np.sin(angles)).flatten()
    V = dipole_potential(x, y)
    triang = mtri.Triangulation(x, y)
    triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1), y[triang.triangles].mean(axis=1)) < min_radius)
    refiner = mtri.UniformTriRefiner(triang)
    tri_refi, z_test_refi = refiner.refine_field(V, subdiv=3)
    tci = mtri.CubicTriInterpolator(triang, -V)
    Ex, Ey = tci.gradient(triang.x, triang.y)
    E_norm = np.hypot(Ex, Ey)
    plt.figure()
    plt.gca().set_aspect('equal')
    plt.triplot(triang, color='0.8')
    levels = np.arange(0.0, 1.0, 0.01)
    cmap = mpl.colormaps['hot']
    plt.tricontour(tri_refi, z_test_refi, levels=levels, cmap=cmap, linewidths=[2.0, 1.0, 1.0, 1.0])
    plt.quiver(triang.x, triang.y, Ex / E_norm, Ey / E_norm, units='xy', scale=10.0, zorder=3, color='blue', width=0.007, headwidth=3.0, headlength=4.0)