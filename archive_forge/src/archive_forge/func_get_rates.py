from __future__ import (absolute_import, division, print_function)
import math
from collections import OrderedDict
from ..util import import_
import numpy as np
from ..symbolic import SymbolicSys, TransformedSys, symmetricsys
def get_rates(x, y, p, be=math, T0=298.15, T0C=273.15, R=8.3144598, kB_over_h=1.38064852e-23 / 6.62607004e-34):
    pd = dict(zip(param_keys, p))
    He_u_T = pd['He_u'] + pd['dCp_u'] * (pd['T'] - T0)
    He_dis_T = pd['He_dis'] + pd['dCp_dis'] * (pd['T'] - T0)
    Se_u = pd['He_u'] / (T0C + pd['Tm_C']) + pd['dCp_u'] * be.log(pd['T'] / T0)
    Se_dis = pd['Se_dis'] + pd['dCp_dis'] * be.log(pd['T'] / T0)

    def C(k):
        return y[names.index(k)]
    return {'unfold': C('N') * Eyring(He_u_T + pd['Ha_f'], pd['Sa_f'] + Se_u, pd['T'], R, kB_over_h, be), 'fold': C('U') * Eyring(pd['Ha_f'], pd['Sa_f'], pd['T'], R, kB_over_h, be), 'aggregate': C('U') * Eyring(pd['Ha_agg'], pd['Sa_agg'], pd['T'], R, kB_over_h, be), 'dissociate': C('NL') * Eyring(He_dis_T + pd['Ha_as'], Se_dis + pd['Sa_as'], pd['T'], R, kB_over_h, be), 'associate': C('N') * C('L') * Eyring(pd['Ha_as'], pd['Sa_as'], pd['T'], R, kB_over_h, be) / molar_unitless}