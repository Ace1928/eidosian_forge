import fractions
import functools
import re
from collections import OrderedDict
from typing import List, Tuple, Dict
import numpy as np
from scipy.spatial import ConvexHull
import ase.units as units
from ase.formula import Formula
def colorfunction(self, U, pH, colors):
    coefs, energy = self.decompose(U, pH, verbose=False)
    indices = tuple(sorted(np.where(abs(coefs) > 0.001)[0]))
    color = colors.get(indices)
    if color is None:
        color = len(colors)
        colors[indices] = color
    return color