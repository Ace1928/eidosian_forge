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
def get_mic_distance(p1, p2, cell, pbc):
    """This method calculates the shortest distance between p1 and p2
    through the cell boundaries defined by cell and pbc.
    This method works for reasonable unit cells, but not for extremely
    elongated ones.
    """
    ct = cell.T
    pos = np.array((p1, p2))
    scaled = np.linalg.solve(ct, pos.T).T
    for i in range(3):
        if pbc[i]:
            scaled[:, i] %= 1.0
            scaled[:, i] %= 1.0
    P = np.dot(scaled, cell)
    pbc_directions = [[-1, 1] * int(direction) + [0] for direction in pbc]
    translations = np.array(list(itertools.product(*pbc_directions))).T
    p0r = np.tile(np.reshape(P[0, :], (3, 1)), (1, translations.shape[1]))
    p1r = np.tile(np.reshape(P[1, :], (3, 1)), (1, translations.shape[1]))
    dp_vec = p0r + np.dot(ct, translations)
    d = np.min(np.power(p1r - dp_vec, 2).sum(axis=0)) ** 0.5
    return d