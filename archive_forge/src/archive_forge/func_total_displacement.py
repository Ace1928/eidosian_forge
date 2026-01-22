import numpy as np
from collections import namedtuple
from ase.geometry import find_mic
def total_displacement(disp):
    disp_a = (disp ** 2).sum(axis=1) ** 0.5
    return sum(disp_a)