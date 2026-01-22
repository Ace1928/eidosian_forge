import re
import warnings
from typing import Dict
import numpy as np
import ase  # Annotations
from ase.utils import jsonable
from ase.cell import Cell
def indices_to_axis_coords(indices, points, cell):
    jump = False
    xcoords = [0]
    for i1, i2 in zip(indices[:-1], indices[1:]):
        if not jump and i1 + 1 == i2:
            length = 0
            jump = True
        else:
            diff = points[i2] - points[i1]
            length = np.linalg.norm(kpoint_convert(cell, skpts_kc=diff))
            jump = False
        xcoords.extend(np.linspace(0, length, i2 - i1 + 1)[1:] + xcoords[-1])
    xcoords = np.array(xcoords)
    return (xcoords, xcoords[indices])