import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_triinterp_transformations():
    n_angles = 20
    n_radii = 10
    min_radius = 0.15

    def z(x, y):
        r1 = np.hypot(0.5 - x, 0.5 - y)
        theta1 = np.arctan2(0.5 - x, 0.5 - y)
        r2 = np.hypot(-x - 0.2, -y - 0.2)
        theta2 = np.arctan2(-x - 0.2, -y - 0.2)
        z = -(2 * (np.exp((r1 / 10) ** 2) - 1) * 30.0 * np.cos(7.0 * theta1) + (np.exp((r2 / 10) ** 2) - 1) * 30.0 * np.cos(11.0 * theta2) + 0.7 * (x ** 2 + y ** 2))
        return (np.max(z) - z) / (np.max(z) - np.min(z))
    radii = np.linspace(min_radius, 0.95, n_radii)
    angles = np.linspace(0 + n_angles, 2 * np.pi + n_angles, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    angles[:, 1::2] += np.pi / n_angles
    x0 = (radii * np.cos(angles)).flatten()
    y0 = (radii * np.sin(angles)).flatten()
    triang0 = mtri.Triangulation(x0, y0)
    z0 = z(x0, y0)
    xs0 = np.linspace(-1.0, 1.0, 23)
    ys0 = np.linspace(-1.0, 1.0, 23)
    xs0, ys0 = np.meshgrid(xs0, ys0)
    xs0 = xs0.ravel()
    ys0 = ys0.ravel()
    interp_z0 = {}
    for i_angle in range(2):
        theta = 2 * np.pi / n_angles * i_angle
        x = np.cos(theta) * x0 + np.sin(theta) * y0
        y = -np.sin(theta) * x0 + np.cos(theta) * y0
        xs = np.cos(theta) * xs0 + np.sin(theta) * ys0
        ys = -np.sin(theta) * xs0 + np.cos(theta) * ys0
        triang = mtri.Triangulation(x, y, triang0.triangles)
        linear_interp = mtri.LinearTriInterpolator(triang, z0)
        cubic_min_E = mtri.CubicTriInterpolator(triang, z0)
        cubic_geom = mtri.CubicTriInterpolator(triang, z0, kind='geom')
        dic_interp = {'lin': linear_interp, 'min_E': cubic_min_E, 'geom': cubic_geom}
        for interp_key in ['lin', 'min_E', 'geom']:
            interp = dic_interp[interp_key]
            if i_angle == 0:
                interp_z0[interp_key] = interp(xs0, ys0)
            else:
                interpz = interp(xs, ys)
                matest.assert_array_almost_equal(interpz, interp_z0[interp_key])
    scale_factor = 987654.321
    for scaled_axis in ('x', 'y'):
        if scaled_axis == 'x':
            x = scale_factor * x0
            y = y0
            xs = scale_factor * xs0
            ys = ys0
        else:
            x = x0
            y = scale_factor * y0
            xs = xs0
            ys = scale_factor * ys0
        triang = mtri.Triangulation(x, y, triang0.triangles)
        linear_interp = mtri.LinearTriInterpolator(triang, z0)
        cubic_min_E = mtri.CubicTriInterpolator(triang, z0)
        cubic_geom = mtri.CubicTriInterpolator(triang, z0, kind='geom')
        dic_interp = {'lin': linear_interp, 'min_E': cubic_min_E, 'geom': cubic_geom}
        for interp_key in ['lin', 'min_E', 'geom']:
            interpz = dic_interp[interp_key](xs, ys)
            matest.assert_array_almost_equal(interpz, interp_z0[interp_key])