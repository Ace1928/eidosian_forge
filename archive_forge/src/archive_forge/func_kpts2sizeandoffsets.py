import os
import copy
import subprocess
from math import pi, sqrt
import pathlib
from typing import Union, Optional, List, Set, Dict, Any
import warnings
import numpy as np
from ase.cell import Cell
from ase.outputs import Properties, all_outputs
from ase.utils import jsonable
from ase.calculators.abc import GetPropertiesMixin
def kpts2sizeandoffsets(size=None, density=None, gamma=None, even=None, atoms=None):
    """Helper function for selecting k-points.

    Use either size or density.

    size: 3 ints
        Number of k-points.
    density: float
        K-point density in units of k-points per Ang^-1.
    gamma: None or bool
        Should the Gamma-point be included?  Yes / no / don't care:
        True / False / None.
    even: None or bool
        Should the number of k-points be even?  Yes / no / don't care:
        True / False / None.
    atoms: Atoms object
        Needed for calculating k-point density.

    """
    if size is not None and density is not None:
        raise ValueError('Cannot specify k-point mesh size and density simultaneously')
    elif density is not None and atoms is None:
        raise ValueError('Cannot set k-points from "density" unless Atoms are provided (need BZ dimensions).')
    if size is None:
        if density is None:
            size = [1, 1, 1]
        else:
            size = kptdensity2monkhorstpack(atoms, density, None)
    if even is not None:
        size = np.array(size)
        remainder = size % 2
        if even:
            size += remainder
        else:
            size += 1 - remainder
    offsets = [0, 0, 0]
    if atoms is None:
        pbc = [True, True, True]
    else:
        pbc = atoms.pbc
    if gamma is not None:
        for i, s in enumerate(size):
            if pbc[i] and s % 2 != bool(gamma):
                offsets[i] = 0.5 / s
    return (size, offsets)