from itertools import chain
from matplotlib import cbook, cm, colors as mcolors, markers, image as mimage
from matplotlib.backends.qt_compat import QtGui
from matplotlib.backends.qt_editor import _formlayout
from matplotlib.dates import DateConverter, num2date
def convert_limits(lim, converter):
    """Convert axis limits for correct input editors."""
    if isinstance(converter, DateConverter):
        return map(num2date, lim)
    return map(float, lim)