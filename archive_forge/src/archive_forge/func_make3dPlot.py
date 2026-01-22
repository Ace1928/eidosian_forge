import logging
from math import pi
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.dependencies.matplotlib import pyplot as plt
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, mcpp_available, MCPP_Error
from pyomo.core import (
from pyomo.core.expr import identify_variables
def make3dPlot(expr, numticks=30, show_plot=False):
    ccSurf = [None] * (numticks + 1) ** 2
    cvSurf = [None] * (numticks + 1) ** 2
    fvals = [None] * (numticks + 1) ** 2
    xaxis2d = [None] * (numticks + 1) ** 2
    yaxis2d = [None] * (numticks + 1) ** 2
    ccAffine = [None] * (numticks + 1) ** 2
    cvAffine = [None] * (numticks + 1) ** 2
    eqn = mc(expr)
    vars = identify_variables(expr)
    x = next(vars)
    y = next(vars)
    x_tick_length = (x.ub - x.lb) / numticks
    y_tick_length = (y.ub - y.lb) / numticks
    xaxis = [x.lb + x_tick_length * n for n in range(numticks + 1)]
    yaxis = [y.lb + y_tick_length * n for n in range(numticks + 1)]
    ccSlope = eqn.subcc()
    cvSlope = eqn.subcv()
    x_val = value(x)
    y_val = value(y)
    f_cc = eqn.concave()
    f_cv = eqn.convex()
    for i, x_tick in enumerate(xaxis):
        eqn.changePoint(x, x_tick)
        for j, y_tick in enumerate(yaxis):
            ccAffine[i + (numticks + 1) * j] = ccSlope[x] * (x_tick - x_val) + ccSlope[y] * (y_tick - y_val) + f_cc
            cvAffine[i + (numticks + 1) * j] = cvSlope[x] * (x_tick - x_val) + cvSlope[y] * (y_tick - y_val) + f_cv
            xaxis2d[i + (numticks + 1) * j] = x_tick
            yaxis2d[i + (numticks + 1) * j] = y_tick
            eqn.changePoint(y, y_tick)
            ccSurf[i + (numticks + 1) * j] = eqn.concave()
            cvSurf[i + (numticks + 1) * j] = eqn.convex()
            fvals[i + (numticks + 1) * j] = value(expr)
    if show_plot:
        from mpl_toolkits.mplot3d import Axes3D
        assert Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(xaxis2d, yaxis2d, cvSurf, color='b')
        ax.scatter(xaxis2d, yaxis2d, fvals, color='r')
        ax.scatter(xaxis2d, yaxis2d, ccSurf, color='b')
        ax.scatter(xaxis2d, yaxis2d, cvAffine, color='k')
        ax.view_init(10, 270)
        plt.show()
    return (ccSurf, cvSurf, ccAffine, cvAffine)