import numpy as np
from pyodesys.util import import_
from pyodesys.core import integrate_chained
from pyodesys.symbolic import SymbolicSys, PartiallySolvedSystem, symmetricsys, TransformedSys
from pyodesys.tests._robertson import get_ode_exprs
def reduced_analytic(x0, y0, p0):
    return {lin_s.dep[reduced - 1]: y0[0] + y0[1] + y0[2] - lin_s.dep[other1] - lin_s.dep[other2]}