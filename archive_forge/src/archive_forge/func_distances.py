from time import time
from math import sqrt, pi
import numpy as np
from ase.parallel import paropen
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.transport.tools import dagger, normalize
from ase.io.jsonio import read_json, write_json
def distances(self, R):
    """Relative distances between centers.

        Returns a matrix with the distances between different Wannier centers.
        R = [n1, n2, n3] is in units of the basis vectors of the small cell
        and allows one to measure the distance with centers moved to a
        different small cell.
        The dimension of the matrix is [Nw, Nw].
        """
    Nw = self.nwannier
    cen = self.get_centers()
    r1 = cen.repeat(Nw, axis=0).reshape(Nw, Nw, 3)
    r2 = cen.copy()
    for i in range(3):
        r2 += self.unitcell_cc[i] * R[i]
    r2 = np.swapaxes(r2.repeat(Nw, axis=0).reshape(Nw, Nw, 3), 0, 1)
    return np.sqrt(np.sum((r1 - r2) ** 2, axis=-1))