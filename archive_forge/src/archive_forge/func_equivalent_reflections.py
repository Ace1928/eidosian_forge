import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
def equivalent_reflections(self, hkl):
    """Return all equivalent reflections to the list of Miller indices
        in hkl.

        Example:

        >>> from ase.spacegroup import Spacegroup
        >>> sg = Spacegroup(225)  # fcc
        >>> sg.equivalent_reflections([[0, 0, 2]])
        array([[ 0,  0, -2],
               [ 0, -2,  0],
               [-2,  0,  0],
               [ 2,  0,  0],
               [ 0,  2,  0],
               [ 0,  0,  2]])
        """
    hkl = np.array(hkl, dtype='int', ndmin=2)
    rot = self.get_rotations()
    n, nrot = (len(hkl), len(rot))
    R = rot.transpose(0, 2, 1).reshape((3 * nrot, 3)).T
    refl = np.dot(hkl, R).reshape((n * nrot, 3))
    ind = np.lexsort(refl.T)
    refl = refl[ind]
    diff = np.diff(refl, axis=0)
    mask = np.any(diff, axis=1)
    return np.vstack((refl[:-1][mask], refl[-1, :]))