from __future__ import annotations
import logging
import warnings
from fractions import Fraction
from functools import reduce
from itertools import chain, combinations, product
from math import cos, floor, gcd
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.fractions import lcm
from numpy.testing import assert_allclose
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.lattice import Lattice
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def fix_pbc(structure, matrix=None):
    """
    Set all frac_coords of the input structure within [0,1].

    Args:
        structure (pymatgen structure object): input structure
        matrix (lattice matrix, 3 by 3 array/matrix): new structure's lattice matrix,
            If None, use input structure's matrix.


    Returns:
        new structure with fixed frac_coords and lattice matrix
    """
    spec = []
    coords = []
    latte = Lattice(structure.lattice.matrix) if matrix is None else Lattice(matrix)
    for site in structure:
        spec.append(site.specie)
        coord = np.array(site.frac_coords)
        for i in range(3):
            coord[i] -= floor(coord[i])
            if np.allclose(coord[i], 1) or np.allclose(coord[i], 0):
                coord[i] = 0
            else:
                coord[i] = round(coord[i], 7)
        coords.append(coord)
    return Structure(latte, spec, coords, site_properties=structure.site_properties)