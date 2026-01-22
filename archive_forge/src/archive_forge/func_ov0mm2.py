from functools import reduce
from itertools import combinations, chain
from math import factorial
from operator import mul
import numpy as np
from ase.units import kg, C, _hbar, kB
from ase.vibrations import Vibrations
def ov0mm2(self, m, delta):
    if m == 0:
        return delta ** 2 / np.sqrt(8) * self.ov0m(m, delta) ** 2
    elif m == 1:
        return delta ** 2 / np.sqrt(8) * (self.ov0m(m, delta) ** 2 - 2 * self.ov0m(m - 1, delta) ** 2)
    else:
        return delta ** 2 / np.sqrt(8) * (self.ov0m(m, delta) ** 2 - 2 * self.ov0m(m - 1, delta) ** 2 + self.ov0m(m - 2, delta) ** 2)