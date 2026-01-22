import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
def symmetry_normalised_sites(self, scaled_positions, map_to_unitcell=True):
    """Returns an array of same size as *scaled_positions*,
        containing the corresponding symmetry-equivalent sites of
        lowest indices.

        If *map_to_unitcell* is true, the returned positions are all
        mapped into the unit cell, i.e. lattice translations are
        included as symmetry operator.

        Example:

        >>> from ase.spacegroup import Spacegroup
        >>> sg = Spacegroup(225)  # fcc
        >>> sg.symmetry_normalised_sites([[0.0, 0.5, 0.5], [1.0, 1.0, 0.0]])
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.]])
        """
    scaled = np.array(scaled_positions, ndmin=2)
    normalised = np.empty(scaled.shape, float)
    rot, trans = self.get_op()
    for i, pos in enumerate(scaled):
        sympos = np.dot(rot, pos) + trans
        if map_to_unitcell:
            sympos %= 1.0
            sympos %= 1.0
        j = np.lexsort(sympos.T)[0]
        normalised[i, :] = sympos[j]
    return normalised