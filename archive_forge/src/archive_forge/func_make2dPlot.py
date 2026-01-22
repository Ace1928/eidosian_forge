import logging
from math import pi
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.dependencies.matplotlib import pyplot as plt
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, mcpp_available, MCPP_Error
from pyomo.core import (
from pyomo.core.expr import identify_variables
def make2dPlot(expr, numticks=10, show_plot=False):
    mc_ccVals = [None] * (numticks + 1)
    mc_cvVals = [None] * (numticks + 1)
    aff_cc = [None] * (numticks + 1)
    aff_cv = [None] * (numticks + 1)
    fvals = [None] * (numticks + 1)
    mc_expr = mc(expr)
    x = next(identify_variables(expr))
    tick_length = (x.ub - x.lb) / numticks
    xaxis = [x.lb + tick_length * n for n in range(numticks + 1)]
    x_val = value(x)
    cc = mc_expr.subcc()
    cv = mc_expr.subcv()
    f_cc = mc_expr.concave()
    f_cv = mc_expr.convex()
    for i, x_tick in enumerate(xaxis):
        aff_cc[i] = cc[x] * (x_tick - x_val) + f_cc
        aff_cv[i] = cv[x] * (x_tick - x_val) + f_cv
        mc_expr.changePoint(x, x_tick)
        mc_ccVals[i] = mc_expr.concave()
        mc_cvVals[i] = mc_expr.convex()
        fvals[i] = value(expr)
    if show_plot:
        plt.plot(xaxis, fvals, 'r', xaxis, mc_ccVals, 'b--', xaxis, mc_cvVals, 'b--', xaxis, aff_cc, 'k|', xaxis, aff_cv, 'k|')
        plt.show()
    return (mc_ccVals, mc_cvVals, aff_cc, aff_cv)