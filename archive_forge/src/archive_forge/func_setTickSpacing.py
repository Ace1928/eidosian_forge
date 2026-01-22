import numpy as np
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui
from .UIGraphicsItem import UIGraphicsItem
def setTickSpacing(self, x=None, y=None):
    """
        Set the grid tick spacing to use.

        Tick spacing for each axis shall be specified as an array of
        descending values, one for each tick scale. When the value
        is set to None, grid line distance is chosen automatically
        for this particular level.

        Example:
            Default setting of 3 scales for each axis:
            setTickSpacing(x=[None, None, None], y=[None, None, None])

            Single scale with distance of 1.0 for X axis, Two automatic
            scales for Y axis:
            setTickSpacing(x=[1.0], y=[None, None])

            Single scale with distance of 1.0 for X axis, Two scales
            for Y axis, one with spacing of 1.0, other one automatic:
            setTickSpacing(x=[1.0], y=[1.0, None])
        """
    self.opts['tickSpacing'] = (x or self.opts['tickSpacing'][0], y or self.opts['tickSpacing'][1])
    self.grid_depth = max([len(s) for s in self.opts['tickSpacing']])
    self.picture = None
    self.update()