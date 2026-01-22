import argparse
import itertools
import numpy as np
from utils import FrameCounter
import pyqtgraph as pg
import pyqtgraph.functions as fn
import pyqtgraph.parametertree as ptree
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
@interactor.decorate(nest=True, nsamples={'limits': [0, None]}, frames={'limits': [1, None]}, fsample={'units': 'Hz'}, frequency={'units': 'Hz'})
def makeData(noise=args.noise, nsamples=args.nsamples, frames=args.frames, fsample=args.fsample, frequency=args.frequency, amplitude=args.amplitude):
    global data, connect_array, ptr
    ttt = np.arange(frames * nsamples, dtype=np.float64) / fsample
    data = amplitude * np.sin(2 * np.pi * frequency * ttt).reshape((frames, nsamples))
    if noise:
        data += np.random.normal(size=data.shape)
    connect_array = np.ones(data.shape[-1], dtype=bool)
    ptr = 0
    pw.setRange(QtCore.QRectF(0, -10, nsamples, 20))