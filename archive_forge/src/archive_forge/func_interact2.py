from math import sqrt, exp, log
import numpy as np
from ase.data import chemical_symbols, atomic_numbers
from ase.units import Bohr
from ase.neighborlist import NeighborList
from ase.calculators.calculator import (Calculator, all_changes,
def interact2(self, a1, a2, d, r, p1, p2, ksi):
    x = exp(self.acut * (r - self.rc))
    theta = 1.0 / (1.0 + x)
    y1 = exp(-p2['eta2'] * (r - beta * p2['s0'])) * ksi / p1['gamma1'] * theta * self.deds[a1]
    y2 = exp(-p1['eta2'] * (r - beta * p1['s0'])) / ksi / p2['gamma1'] * theta * self.deds[a2]
    f = (y1 * p2['eta2'] + y2 * p1['eta2'] + (y1 + y2) * self.acut * theta * x) * d / r
    self.forces[a1] -= f
    self.forces[a2] += f
    self.stress += np.outer(f, d)