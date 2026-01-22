from functools import reduce
from itertools import combinations, chain
from math import factorial
from operator import mul
import numpy as np
from ase.units import kg, C, _hbar, kB
from ase.vibrations import Vibrations
def direct0mm1(self, m, delta):
    """direct and fast <0|m><m|1>"""
    S = delta ** 2 / 2.0
    sum = S ** m
    if m:
        sum -= m * S ** (m - 1)
    return np.where(S == 0, 0, np.exp(-S) * delta / np.sqrt(2) * sum * self.factorial.inv(m))