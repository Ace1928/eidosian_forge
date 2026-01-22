import collections
import os
import sys
from time import perf_counter
import numpy as np
import pyqtgraph as pg
from pyqtgraph import configfile
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree import types as pTypes
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
def runInertial(self, nPts):
    clocks = self.clocks
    dt = self.dt
    tVals = np.linspace(0, dt * (nPts - 1), nPts)
    for cl in self.clocks.values():
        for i in range(1, nPts):
            nextT = tVals[i]
            while True:
                tau1, tau2 = cl.accelLimits()
                x = cl.x
                v = cl.v
                tau = cl.pt
                g = cl.acceleration()
                v1, x1, tau1 = self.hypTStep(dt, v, x, tau, g)
                if tau1 > tau2:
                    dtau = tau2 - tau
                    cl.v, cl.x, cl.t = self.tauStep(dtau, v, x, cl.t, g)
                    cl.pt = tau2
                else:
                    cl.v, cl.x, cl.pt = (v1, x1, tau1)
                    cl.t += dt
                if cl.t >= nextT:
                    cl.refx = cl.x
                    cl.refv = cl.v
                    cl.reft = cl.t
                    cl.recordFrame(i)
                    break