from __future__ import annotations
import itertools
import logging
import warnings
from typing import TYPE_CHECKING
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from pymatgen.core.structure import Structure
from pymatgen.util.coord import get_angle
from pymatgen.util.string import unicodeify_spacegroup
@property
def total_surface_energy(self) -> float:
    """
        Total surface energy of the Wulff shape.

        Returns:
            float: sum(surface_energy_hkl * area_hkl)
        """
    tot_surface_energy = 0.0
    for hkl, energy in self.miller_energy_dict.items():
        tot_surface_energy += energy * self.miller_area_dict[hkl]
    return tot_surface_energy