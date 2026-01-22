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
def closest_distances_generator(atom_numbers, ratio_of_covalent_radii):
    """Generates the blmin dict used across the GA.
    The distances are based on the covalent radii of the atoms.
    """
    cr = covalent_radii
    ratio = ratio_of_covalent_radii
    blmin = dict()
    for i in atom_numbers:
        blmin[i, i] = cr[i] * 2 * ratio
        for j in atom_numbers:
            if i == j:
                continue
            if (i, j) in blmin.keys():
                continue
            blmin[i, j] = blmin[j, i] = ratio * (cr[i] + cr[j])
    return blmin