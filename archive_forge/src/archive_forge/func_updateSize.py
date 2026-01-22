import argparse
import itertools
import sys
import numpy as np
from utils import FrameCounter
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import VideoTemplate_generic as ui_template
def updateSize():
    global ui, vb
    frames = ui.framesSpin.value()
    width = ui.widthSpin.value()
    height = ui.heightSpin.value()
    dtype = xp.dtype(str(ui.dtypeCombo.currentText()))
    rgb = 3 if ui.rgbCheck.isChecked() else 1
    ui.sizeLabel.setText('%d MB' % (frames * width * height * rgb * dtype.itemsize / 1000000.0))
    vb.setRange(QtCore.QRectF(0, 0, width, height))