import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
def unique_sites(self, scaled_positions, symprec=0.001, output_mask=False, map_to_unitcell=True):
    """Returns a subset of *scaled_positions* containing only the
        symmetry-unique positions.  If *output_mask* is True, a boolean
        array masking the subset is also returned.

        If *map_to_unitcell* is true, all sites are first mapped into
        the unit cell making e.g. [0, 0, 0] and [1, 0, 0] equivalent.

        Example:

        >>> from ase.spacegroup import Spacegroup
        >>> sg = Spacegroup(225)  # fcc
        >>> sg.unique_sites([[0.0, 0.0, 0.0],
        ...                  [0.5, 0.5, 0.0],
        ...                  [1.0, 0.0, 0.0],
        ...                  [0.5, 0.0, 0.0]])
        array([[ 0. ,  0. ,  0. ],
               [ 0.5,  0. ,  0. ]])
        """
    scaled = np.array(scaled_positions, ndmin=2)
    symnorm = self.symmetry_normalised_sites(scaled, map_to_unitcell)
    perm = np.lexsort(symnorm.T)
    iperm = perm.argsort()
    xmask = np.abs(np.diff(symnorm[perm], axis=0)).max(axis=1) > symprec
    mask = np.concatenate(([True], xmask))
    imask = mask[iperm]
    if output_mask:
        return (scaled[imask], imask)
    else:
        return scaled[imask]