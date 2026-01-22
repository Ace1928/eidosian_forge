from __future__ import annotations
import collections
import itertools
from math import acos, pi
from typing import TYPE_CHECKING
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
from pymatgen.analysis.local_env import JmolNN, VoronoiNN
from pymatgen.core import Composition, Element, PeriodicSite, Species
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_percentage_lattice_parameter_changes(self) -> dict[str, float]:
    """
        Returns the percentage lattice parameter changes.

        Returns:
            dict[str, float]: Percent changes in lattice parameter, e.g.,
                {'a': 0.012, 'b': 0.021, 'c': -0.031} implies a change of 1.2%,
                2.1% and -3.1% in the a, b and c lattice parameters respectively.
        """
    initial_latt = self.initial.lattice
    final_latt = self.final.lattice
    return {length: getattr(final_latt, length) / getattr(initial_latt, length) - 1 for length in ['a', 'b', 'c']}