import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
from mpl_toolkits.axisartist import AxisArtistHelperRectlinear
from mpl_toolkits.axisartist.axis_artist import (AxisArtist, AxisLabel,
@image_comparison(['axis_artist_ticklabels.png'], style='default')
def test_ticklabels():
    plt.rcParams['text.kerning_factor'] = 6
    fig, ax = plt.subplots()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.plot([0.2, 0.4], [0.5, 0.5], 'o')
    ticks = Ticks(ticksize=10, axis=ax.xaxis)
    ax.add_artist(ticks)
    locs_angles_labels = [((0.2, 0.5), -90, '0.2'), ((0.4, 0.5), -120, '0.4')]
    tick_locs_angles = [(xy, a + 180) for xy, a, l in locs_angles_labels]
    ticks.set_locs_angles(tick_locs_angles)
    ticklabels = TickLabels(axis_direction='left')
    ticklabels._locs_angles_labels = locs_angles_labels
    ticklabels.set_pad(10)
    ax.add_artist(ticklabels)
    ax.plot([0.5], [0.5], 's')
    axislabel = AxisLabel(0.5, 0.5, 'Test')
    axislabel._offset_radius = 20
    axislabel._ref_angle = 0
    axislabel.set_axis_direction('bottom')
    ax.add_artist(axislabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)