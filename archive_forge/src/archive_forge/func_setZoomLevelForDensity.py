import sys
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import numpy as np
from .AxisItem import AxisItem
def setZoomLevelForDensity(self, density):
    """
        Setting `zoomLevel` and `minSpacing` based on given density of seconds per pixel
        
        The display format is adjusted automatically depending on the current time
        density (seconds/point) on the axis. You can customize the behaviour by 
        overriding this function or setting a different set of zoom levels
        than the default one. The `zoomLevels` variable is a dictionary with the
        maximal distance of ticks in seconds which are allowed for each zoom level
        before the axis switches to the next coarser level. To customize the zoom level
        selection, override this function.
        """
    padding = 10
    if self.orientation in ['bottom', 'top']:

        def sizeOf(text):
            return self.fontMetrics.boundingRect(text).width() + padding
    else:

        def sizeOf(text):
            return self.fontMetrics.boundingRect(text).height() + padding
    self.zoomLevel = YEAR_MONTH_ZOOM_LEVEL
    for maximalSpacing, zoomLevel in self.zoomLevels.items():
        size = sizeOf(zoomLevel.exampleText)
        if maximalSpacing / size < density:
            break
        self.zoomLevel = zoomLevel
    self.zoomLevel.utcOffset = self.utcOffset
    size = sizeOf(self.zoomLevel.exampleText)
    self.minSpacing = density * size