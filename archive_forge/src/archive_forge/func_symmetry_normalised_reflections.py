import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
def symmetry_normalised_reflections(self, hkl):
    """Returns an array of same size as *hkl*, containing the
        corresponding symmetry-equivalent reflections of lowest
        indices.

        Example:

        >>> from ase.spacegroup import Spacegroup
        >>> sg = Spacegroup(225)  # fcc
        >>> sg.symmetry_normalised_reflections([[2, 0, 0], [0, 2, 0]])
        array([[ 0,  0, -2],
               [ 0,  0, -2]])
        """
    hkl = np.array(hkl, dtype=int, ndmin=2)
    normalised = np.empty(hkl.shape, int)
    R = self.get_rotations().transpose(0, 2, 1)
    for i, g in enumerate(hkl):
        gsym = np.dot(R, g)
        j = np.lexsort(gsym.T)[0]
        normalised[i, :] = gsym[j]
    return normalised