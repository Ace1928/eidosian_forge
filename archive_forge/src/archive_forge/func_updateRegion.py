import numpy as np
import pyqtgraph as pg
def updateRegion(window, viewRange):
    rgn = viewRange[0]
    region.setRegion(rgn)