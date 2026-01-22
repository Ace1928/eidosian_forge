import numpy as np
import itertools
from scipy import sparse as sp
from scipy.spatial import cKDTree
import scipy.sparse.csgraph as csgraph
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import complete_cell, find_mic, wrap_positions
from ase.geometry import minkowski_reduce
from ase.cell import Cell
def build_neighbor_list(atoms, cutoffs=None, **kwargs):
    """Automatically build and update a NeighborList.

    Parameters:

    atoms : :class:`~ase.Atoms` object
        Atoms to build Neighborlist for.
    cutoffs: list of floats
        Radii for each atom. If not given it will be produced by calling :func:`ase.neighborlist.natural_cutoffs`
    kwargs: arbitrary number of options
        Will be passed to the constructor of :class:`~ase.neighborlist.NeighborList`

    Returns:

    return: :class:`~ase.neighborlist.NeighborList`
        A :class:`~ase.neighborlist.NeighborList` instance (updated).
    """
    if cutoffs is None:
        cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, **kwargs)
    nl.update(atoms)
    return nl