import numpy as np
import itertools
from scipy import sparse as sp
from scipy.spatial import cKDTree
import scipy.sparse.csgraph as csgraph
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import complete_cell, find_mic, wrap_positions
from ase.geometry import minkowski_reduce
from ase.cell import Cell
def first_neighbors(natoms, first_atom):
    """
    Compute an index array pointing to the ranges within the neighbor list that
    contain the neighbors for a certain atom.

    Parameters:

    natoms : int
        Total number of atom.
    first_atom : array_like
        Array containing the first atom 'i' of the neighbor tuple returned
        by the neighbor list.

    Returns:

    seed : array
        Array containing pointers to the start and end location of the
        neighbors of a certain atom. Neighbors of atom k have indices from s[k]
        to s[k+1]-1.
    """
    if len(first_atom) == 0:
        return np.zeros(natoms + 1, dtype=int)
    seed = -np.ones(natoms + 1, dtype=int)
    first_atom = np.asarray(first_atom)
    mask = first_atom[:-1] != first_atom[1:]
    seed[first_atom[0]] = 0
    seed[-1] = len(first_atom)
    seed[first_atom[1:][mask]] = (np.arange(len(mask)) + 1)[mask]
    mask = seed == -1
    while mask.any():
        seed[mask] = seed[np.arange(natoms + 1)[mask] + 1]
        mask = seed == -1
    return seed