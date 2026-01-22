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
@property
def sigma_from_site_prop(self) -> int:
    """
        This method returns the sigma value of the GB from site properties.
        If the GB structure merge some atoms due to the atoms too closer with
        each other, this property will not work.
        """
    n_coi = 0
    if None in self.site_properties['grain_label']:
        raise RuntimeError('Site were merged, this property do not work')
    for tag in self.site_properties['grain_label']:
        if 'incident' in tag:
            n_coi += 1
    return int(round(len(self) / n_coi))