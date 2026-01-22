import itertools
import numpy as np
from ase.geometry import complete_cell
from ase.geometry.minkowski_reduction import minkowski_reduce
from ase.utils import pbc2pbc
from ase.cell import Cell
def get_duplicate_atoms(atoms, cutoff=0.1, delete=False):
    """Get list of duplicate atoms and delete them if requested.

    Identify all atoms which lie within the cutoff radius of each other.
    Delete one set of them if delete == True.
    """
    from scipy.spatial.distance import pdist
    dists = pdist(atoms.get_positions(), 'sqeuclidean')
    dup = np.nonzero(dists < cutoff ** 2)
    rem = np.array(_row_col_from_pdist(len(atoms), dup[0]))
    if delete:
        if rem.size != 0:
            del atoms[rem[:, 0]]
    else:
        return rem