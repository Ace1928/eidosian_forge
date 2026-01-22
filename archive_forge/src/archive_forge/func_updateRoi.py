import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
def updateRoi(roi):
    global im1, im2, im3, im4, arr, lastRoi
    if roi is None:
        return
    lastRoi = roi
    arr1 = roi.getArrayRegion(im1.image, img=im1)
    im3.setImage(arr1)
    arr2 = roi.getArrayRegion(im2.image, img=im2)
    im4.setImage(arr2)
    updateRoiPlot(roi, arr1)