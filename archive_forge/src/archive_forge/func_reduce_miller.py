from typing import List, Optional
import numpy as np
from ase.data import atomic_numbers as ref_atomic_numbers
from ase.spacegroup import Spacegroup
from ase.cluster.base import ClusterBase
from ase.cluster.cluster import Cluster
def reduce_miller(hkl):
    """Reduce Miller index to the lowest equivalent integers."""
    hkl = np.array(hkl)
    old = hkl.copy()
    d = GCD(GCD(hkl[0], hkl[1]), hkl[2])
    while d != 1:
        hkl = hkl // d
        d = GCD(GCD(hkl[0], hkl[1]), hkl[2])
    if np.dot(old, hkl) > 0:
        return hkl
    else:
        return -hkl