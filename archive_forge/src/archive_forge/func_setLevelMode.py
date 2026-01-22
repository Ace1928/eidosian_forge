import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
def setLevelMode():
    mode = 'mono' if monoRadio.isChecked() else 'rgba'
    hist.setLevelMode(mode)