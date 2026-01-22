import inspect
import json
import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import atoms_too_close, gather_atoms_by_tag
from scipy.spatial.distance import cdist
def get_number_of_valence_electrons(Z):
    """Return the number of valence electrons for the element with
    atomic number Z, simply based on its periodic table group.
    """
    groups = [[], [1, 3, 11, 19, 37, 55, 87], [2, 4, 12, 20, 38, 56, 88], [21, 39, 57, 89]]
    for i in range(9):
        groups.append(i + np.array([22, 40, 72, 104]))
    for i in range(6):
        groups.append(i + np.array([5, 13, 31, 49, 81, 113]))
    for i, group in enumerate(groups):
        if Z in group:
            nval = i if i < 13 else i - 10
            break
    else:
        raise ValueError('Z=%d not included in this dataset.' % Z)
    return nval