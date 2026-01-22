import itertools
import numpy as np
from ase.geometry import complete_cell
from ase.geometry.minkowski_reduction import minkowski_reduce
from ase.utils import pbc2pbc
from ase.cell import Cell
def conditional_find_mic(vectors, cell, pbc):
    """Return list of vector arrays and corresponding list of vector lengths
    for a given list of vector arrays. The minimum image convention is applied
    if cell and pbc are set. Can be used like a simple version of get_distances.
    """
    if (cell is None) != (pbc is None):
        raise ValueError('cell or pbc must be both set or both be None')
    if cell is not None:
        mics = [find_mic(v, cell, pbc) for v in vectors]
        vectors, vector_lengths = zip(*mics)
    else:
        vector_lengths = np.linalg.norm(vectors, axis=2)
    return ([np.asarray(v) for v in vectors], vector_lengths)