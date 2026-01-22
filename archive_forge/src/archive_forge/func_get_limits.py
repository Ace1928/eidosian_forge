import numpy as np
from ase.calculators.calculator import Calculator
from ase.utils import ff
def get_limits(indices):
    gstarts = []
    gstops = []
    lstarts = []
    lstops = []
    for l, g in enumerate(indices):
        g3, l3 = (3 * g, 3 * l)
        gstarts.append(g3)
        gstops.append(g3 + 3)
        lstarts.append(l3)
        lstops.append(l3 + 3)
    return zip(gstarts, gstops, lstarts, lstops)