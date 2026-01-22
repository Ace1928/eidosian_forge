import os
import time
import math
import itertools
import numpy as np
from scipy.spatial.distance import cdist
from ase.io import write, read
from ase.geometry.cell import cell_to_cellpar
from ase.data import covalent_radii
from ase.ga import get_neighbor_list
def get_cell_angles_lengths(cell):
    """Returns cell vectors lengths (a,b,c) as well as different
    angles (alpha, beta, gamma, phi, chi, psi) (in radians).
    """
    cellpar = cell_to_cellpar(cell)
    cellpar[3:] *= np.pi / 180
    parnames = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
    values = {n: p for n, p in zip(parnames, cellpar)}
    volume = abs(np.linalg.det(cell))
    for i, param in enumerate(['phi', 'chi', 'psi']):
        ab = np.linalg.norm(np.cross(cell[(i + 1) % 3, :], cell[(i + 2) % 3, :]))
        c = np.linalg.norm(cell[i, :])
        s = np.abs(volume / (ab * c))
        if 1 + 1e-06 > s > 1:
            s = 1.0
        values[param] = np.arcsin(s)
    return values