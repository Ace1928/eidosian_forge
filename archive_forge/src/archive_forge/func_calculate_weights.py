from time import time
from math import sqrt, pi
import numpy as np
from ase.parallel import paropen
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.transport.tools import dagger, normalize
from ase.io.jsonio import read_json, write_json
def calculate_weights(cell_cc, normalize=True):
    """ Weights are used for non-cubic cells, see PRB **61**, 10040"""
    alldirs_dc = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=int)
    g = np.dot(cell_cc, cell_cc.T)
    w = np.zeros(6)
    w[0] = g[0, 0] - g[0, 1] - g[0, 2]
    w[1] = g[1, 1] - g[0, 1] - g[1, 2]
    w[2] = g[2, 2] - g[0, 2] - g[1, 2]
    w[3] = g[0, 1]
    w[4] = g[0, 2]
    w[5] = g[1, 2]
    Gdir_dc = alldirs_dc[:3]
    weight_d = w[:3]
    for d in range(3, 6):
        if abs(w[d]) > 1e-05:
            Gdir_dc = np.concatenate((Gdir_dc, alldirs_dc[d:d + 1]))
            weight_d = np.concatenate((weight_d, w[d:d + 1]))
    if normalize:
        weight_d /= max(abs(weight_d))
    return (weight_d, Gdir_dc)