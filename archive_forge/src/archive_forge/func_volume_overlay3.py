from __future__ import (absolute_import, division, print_function,
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import TICKLEFT, TICKRIGHT, Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from six.moves import xrange, zip
def volume_overlay3(ax, quotes, colorup='k', colordown='r', width=4, alpha=1.0):
    """Add a volume overlay to the current axes.  quotes is a list of (d,
    open, high, low, close, volume) and close-open is used to
    determine the color of the bar

    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    quotes : sequence of (time, open, high, low, close, ...) sequences
        data to plot.  time must be in float date format - see date2num
    width : int
        the bar width in points
    colorup : color
        the color of the lines where close1 >= close0
    colordown : color
        the color of the lines where close1 <  close0
    alpha : float
         bar transparency

    Returns
    -------
    ret : `barCollection`
        The `barrCollection` added to the axes


    """
    colorup = mcolors.to_rgba(colorup, alpha)
    colordown = mcolors.to_rgba(colordown, alpha)
    colord = {True: colorup, False: colordown}
    dates, opens, highs, lows, closes, volumes = list(zip(*quotes))
    colors = [colord[close1 >= close0] for close0, close1 in zip(closes[:-1], closes[1:]) if close0 != -1 and close1 != -1]
    colors.insert(0, colord[closes[0] >= opens[0]])
    right = width / 2.0
    left = -width / 2.0
    bars = [((left, 0), (left, volume), (right, volume), (right, 0)) for d, open, high, low, close, volume in quotes]
    sx = ax.figure.dpi * (1.0 / 72.0)
    sy = ax.bbox.height / ax.viewLim.height
    barTransform = Affine2D().scale(sx, sy)
    dates = [d for d, open, high, low, close, volume in quotes]
    offsetsBars = [(d, 0) for d in dates]
    useAA = (0,)
    lw = (0.5,)
    barCollection = PolyCollection(bars, facecolors=colors, edgecolors=((0, 0, 0, 1),), antialiaseds=useAA, linewidths=lw, offsets=offsetsBars, transOffset=ax.transData)
    barCollection.set_transform(barTransform)
    minpy, maxx = (min(dates), max(dates))
    miny = 0
    maxy = max([volume for d, open, high, low, close, volume in quotes])
    corners = ((minpy, miny), (maxx, maxy))
    ax.update_datalim(corners)
    ax.add_collection(barCollection)
    ax.autoscale_view()
    return barCollection