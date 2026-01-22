import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
def equivalent_sites(self, scaled_positions, onduplicates='error', symprec=0.001, occupancies=None):
    """Returns the scaled positions and all their equivalent sites.

        Parameters:

        scaled_positions: list | array
            List of non-equivalent sites given in unit cell coordinates.

        occupancies: list | array, optional (default=None)
            List of occupancies corresponding to the respective sites.

        onduplicates : 'keep' | 'replace' | 'warn' | 'error'
            Action if `scaled_positions` contain symmetry-equivalent
            positions of full occupancy:

            'keep'
               ignore additional symmetry-equivalent positions
            'replace'
                replace
            'warn'
                like 'keep', but issue an UserWarning
            'error'
                raises a SpacegroupValueError

        symprec: float
            Minimum "distance" betweed two sites in scaled coordinates
            before they are counted as the same site.

        Returns:

        sites: array
            A NumPy array of equivalent sites.
        kinds: list
            A list of integer indices specifying which input site is
            equivalent to the corresponding returned site.

        Example:

        >>> from ase.spacegroup import Spacegroup
        >>> sg = Spacegroup(225)  # fcc
        >>> sites, kinds = sg.equivalent_sites([[0, 0, 0], [0.5, 0.0, 0.0]])
        >>> sites
        array([[ 0. ,  0. ,  0. ],
               [ 0. ,  0.5,  0.5],
               [ 0.5,  0. ,  0.5],
               [ 0.5,  0.5,  0. ],
               [ 0.5,  0. ,  0. ],
               [ 0. ,  0.5,  0. ],
               [ 0. ,  0. ,  0.5],
               [ 0.5,  0.5,  0.5]])
        >>> kinds
        [0, 0, 0, 0, 1, 1, 1, 1]
        """
    kinds = []
    sites = []
    scaled = np.array(scaled_positions, ndmin=2)
    for kind, pos in enumerate(scaled):
        for rot, trans in self.get_symop():
            site = np.mod(np.dot(rot, pos) + trans, 1.0)
            if not sites:
                sites.append(site)
                kinds.append(kind)
                continue
            t = site - sites
            mask = np.all((abs(t) < symprec) | (abs(abs(t) - 1.0) < symprec), axis=1)
            if np.any(mask):
                inds = np.argwhere(mask).flatten()
                for ind in inds:
                    if kinds[ind] == kind:
                        pass
                    elif onduplicates == 'keep':
                        pass
                    elif onduplicates == 'replace':
                        kinds[ind] = kind
                    elif onduplicates == 'warn':
                        warnings.warn('scaled_positions %d and %d are equivalent' % (kinds[ind], kind))
                    elif onduplicates == 'error':
                        raise SpacegroupValueError('scaled_positions %d and %d are equivalent' % (kinds[ind], kind))
                    else:
                        raise SpacegroupValueError('Argument "onduplicates" must be one of: "keep", "replace", "warn" or "error".')
            else:
                sites.append(site)
                kinds.append(kind)
    return (np.array(sites), kinds)