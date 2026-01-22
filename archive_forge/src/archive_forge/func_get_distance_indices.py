import numpy as np
import itertools
from scipy import sparse as sp
from scipy.spatial import cKDTree
import scipy.sparse.csgraph as csgraph
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import complete_cell, find_mic, wrap_positions
from ase.geometry import minkowski_reduce
from ase.cell import Cell
def get_distance_indices(distanceMatrix, distance):
    """Get indices for each node that are distance or less away.

    Parameters:

    distanceMatrix: any one of scipy.sparse matrices (NxN)
        Matrix containing distance information of atoms. Get it e.g. with
        :func:`~ase.neighborlist.get_distance_matrix`
    distance: integer
        Number of steps to allow.

    Returns:

    return: list of length N
        A list of length N. return[i] has all indices that are connected to item i.

    The distance matrix only contains shortest paths, so when looking for
    distances longer than one, we need to add the lower values for cases
    where atoms are connected via a shorter path too.
    """
    shape = distanceMatrix.get_shape()
    indices = []
    for i in range(shape[0]):
        row = distanceMatrix.getrow(i)[0]
        found = sp.find(row)
        equal = np.where(found[-1] <= distance)[0]
        indices.append([found[1][x] for x in equal])
    return indices