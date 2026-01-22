import itertools
import numpy as np
from ase.geometry import complete_cell
from ase.geometry.minkowski_reduction import minkowski_reduce
from ase.utils import pbc2pbc
from ase.cell import Cell
def find_mic(v, cell, pbc=True):
    """Finds the minimum-image representation of vector(s) v using either one
    of two find mic algorithms depending on the given cell, v and pbc."""
    cell = Cell(cell)
    pbc = cell.any(1) & pbc2pbc(pbc)
    dim = np.sum(pbc)
    v = np.asarray(v)
    single = v.ndim == 1
    v = np.atleast_2d(v)
    if dim > 0:
        naive_find_mic_is_safe = False
        if dim == 3:
            vmin, vlen = naive_find_mic(v, cell)
            if (vlen < 0.5 * min(cell.lengths())).all():
                naive_find_mic_is_safe = True
        if not naive_find_mic_is_safe:
            vmin, vlen = general_find_mic(v, cell, pbc=pbc)
    else:
        vmin = v.copy()
        vlen = np.linalg.norm(vmin, axis=1)
    if single:
        return (vmin[0], vlen[0])
    else:
        return (vmin, vlen)