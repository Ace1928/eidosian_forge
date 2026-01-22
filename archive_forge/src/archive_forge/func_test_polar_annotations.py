import numpy as np
from numpy.testing import assert_allclose
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison, check_figures_equal
@image_comparison(['polar_axes'], style='default', tol=0.012)
def test_polar_annotations():
    r = np.arange(0.0, 1.0, 0.001)
    theta = 2.0 * 2.0 * np.pi * r
    fig = plt.figure()
    ax = fig.add_subplot(polar=True)
    line, = ax.plot(theta, r, color='#ee8d18', lw=3)
    line, = ax.plot((0, 0), (0, 1), color='#0000ff', lw=1)
    ind = 800
    thisr, thistheta = (r[ind], theta[ind])
    ax.plot([thistheta], [thisr], 'o')
    ax.annotate('a polar annotation', xy=(thistheta, thisr), xytext=(0.05, 0.05), textcoords='figure fraction', arrowprops=dict(facecolor='black', shrink=0.05), horizontalalignment='left', verticalalignment='baseline')
    ax.tick_params(axis='x', tick1On=True, tick2On=True, direction='out')