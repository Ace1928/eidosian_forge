import os
import sys
import h5py
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
def setHDF5(self, data):
    self.hdf5 = data
    self.updateHDF5Plot()