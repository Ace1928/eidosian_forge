from functools import reduce
from itertools import combinations, chain
from math import factorial
from operator import mul
import numpy as np
from ase.units import kg, C, _hbar, kB
from ase.vibrations import Vibrations
def ov0m(self, m, delta):
    if m == 0:
        return np.exp(-0.25 * delta ** 2)
    else:
        assert m > 0
        return -delta / np.sqrt(2 * m) * self.ov0m(m - 1, delta)