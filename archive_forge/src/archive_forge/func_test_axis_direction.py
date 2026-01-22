import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D, Transform
from matplotlib.testing.decorators import image_comparison
from mpl_toolkits.axisartist import SubplotHost
from mpl_toolkits.axes_grid1.parasite_axes import host_axes_class_factory
from mpl_toolkits.axisartist import angle_helper
from mpl_toolkits.axisartist.axislines import Axes
from mpl_toolkits.axisartist.grid_helper_curvelinear import \
@image_comparison(['axis_direction.png'], style='default', tol=0.12)
def test_axis_direction():
    plt.rcParams['text.kerning_factor'] = 6
    fig = plt.figure(figsize=(5, 5))
    tr = Affine2D().scale(np.pi / 180.0, 1.0) + PolarAxes.PolarTransform()
    extreme_finder = angle_helper.ExtremeFinderCycle(20, 20, lon_cycle=360, lat_cycle=None, lon_minmax=None, lat_minmax=(0, np.inf))
    grid_locator1 = angle_helper.LocatorDMS(12)
    tick_formatter1 = angle_helper.FormatterDMS()
    grid_helper = GridHelperCurveLinear(tr, extreme_finder=extreme_finder, grid_locator1=grid_locator1, tick_formatter1=tick_formatter1)
    ax1 = SubplotHost(fig, 1, 1, 1, grid_helper=grid_helper)
    for axis in ax1.axis.values():
        axis.set_visible(False)
    fig.add_subplot(ax1)
    ax1.axis['lat1'] = axis = grid_helper.new_floating_axis(0, 130, axes=ax1, axis_direction='left')
    axis.label.set_text('Test')
    axis.label.set_visible(True)
    axis.get_helper().set_extremes(0.001, 10)
    ax1.axis['lat2'] = axis = grid_helper.new_floating_axis(0, 50, axes=ax1, axis_direction='right')
    axis.label.set_text('Test')
    axis.label.set_visible(True)
    axis.get_helper().set_extremes(0.001, 10)
    ax1.axis['lon'] = axis = grid_helper.new_floating_axis(1, 10, axes=ax1, axis_direction='bottom')
    axis.label.set_text('Test 2')
    axis.get_helper().set_extremes(50, 130)
    axis.major_ticklabels.set_axis_direction('top')
    axis.label.set_axis_direction('top')
    grid_helper.grid_finder.grid_locator1.set_params(nbins=5)
    grid_helper.grid_finder.grid_locator2.set_params(nbins=5)
    ax1.set_aspect(1.0)
    ax1.set_xlim(-8, 8)
    ax1.set_ylim(-4, 12)
    ax1.grid(True)