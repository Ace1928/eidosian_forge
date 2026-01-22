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
def get_atoms_distribution(atoms, number_of_bins=5, max_distance=8, center=None, no_count_types=None):
    """Method to get the distribution of atoms in the
    structure in bins of distances from a defined
    center. Option added to remove counting of
    certain atom types.
    """
    pbc = atoms.get_pbc()
    cell = atoms.get_cell()
    if center is None:
        cx = sum(cell[:, 0]) / 2.0
        cy = sum(cell[:, 1]) / 2.0
        cz = sum(cell[:, 2]) / 2.0
        center = (cx, cy, cz)
    bins = [0] * number_of_bins
    if no_count_types is None:
        no_count_types = []
    for atom in atoms:
        if atom.number not in no_count_types:
            d = get_mic_distance(atom.position, center, cell, pbc)
            for k in range(number_of_bins - 1):
                min_dis_cur_bin = k * max_distance / (number_of_bins - 1.0)
                max_dis_cur_bin = (k + 1) * max_distance / (number_of_bins - 1.0)
                if min_dis_cur_bin < d < max_dis_cur_bin:
                    bins[k] += 1
            if d > max_distance:
                bins[number_of_bins - 1] += 1
    return bins