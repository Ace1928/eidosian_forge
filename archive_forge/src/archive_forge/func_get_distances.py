import itertools
import numpy as np
from ase.geometry import complete_cell
from ase.geometry.minkowski_reduction import minkowski_reduce
from ase.utils import pbc2pbc
from ase.cell import Cell
def get_distances(p1, p2=None, cell=None, pbc=None):
    """Return distance matrix of every position in p1 with every position in p2

    If p2 is not set, it is assumed that distances between all positions in p1
    are desired. p2 will be set to p1 in this case.

    Use set cell and pbc to use the minimum image convention.
    """
    p1 = np.atleast_2d(p1)
    if p2 is None:
        np1 = len(p1)
        ind1, ind2 = np.triu_indices(np1, k=1)
        D = p1[ind2] - p1[ind1]
    else:
        p2 = np.atleast_2d(p2)
        D = (p2[np.newaxis, :, :] - p1[:, np.newaxis, :]).reshape((-1, 3))
    (D,), (D_len,) = conditional_find_mic([D], cell=cell, pbc=pbc)
    if p2 is None:
        Dout = np.zeros((np1, np1, 3))
        Dout[ind1, ind2] = D
        Dout -= np.transpose(Dout, axes=(1, 0, 2))
        Dout_len = np.zeros((np1, np1))
        Dout_len[ind1, ind2] = D_len
        Dout_len += Dout_len.T
        return (Dout, Dout_len)
    D.shape = (-1, len(p2), 3)
    D_len.shape = (-1, len(p2))
    return (D, D_len)