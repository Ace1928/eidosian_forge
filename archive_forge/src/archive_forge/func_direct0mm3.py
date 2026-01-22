from functools import reduce
from itertools import combinations, chain
from math import factorial
from operator import mul
import numpy as np
from ase.units import kg, C, _hbar, kB
from ase.vibrations import Vibrations
def direct0mm3(self, m, delta):
    S = delta ** 2 / 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(S == 0, 0, np.exp(-S) * S ** (m - 1) / delta * np.sqrt(12.0) * (S ** 3 / 6.0 - m * S ** 2 / 2 + m * (m - 1) * S / 2.0 - m * (m - 1) * (m - 2) / 6) * self.factorial.inv(m))