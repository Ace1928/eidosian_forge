from __future__ import annotations
import copy
import itertools
import random
import warnings
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol
from sympy.solvers import linsolve, solve
from pymatgen.analysis.wulff import WulffShape
from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.surface import get_slab_regions
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.outputs import Locpot, Outcar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
def is_converged(self, min_points_frac=0.015, tol: float=0.0025):
    """
        A well converged work function should have a flat electrostatic
            potential within some distance (min_point) about where the peak
            electrostatic potential is found along the c direction of the
            slab. This is dependent on the size of the slab.

        Args:
            min_point (fractional coordinates): The number of data points
                +/- the point of where the electrostatic potential is at
                its peak along the c direction.
            tol (float): If the electrostatic potential stays the same
                within this tolerance, within the min_points, it is converged.

        Returns a bool (whether or not the work function is converged)
        """
    conv_within = tol * (max(self.locpot_along_c) - min(self.locpot_along_c))
    min_points = int(min_points_frac * len(self.locpot_along_c))
    peak_i = self.locpot_along_c.index(self.vacuum_locpot)
    all_flat = []
    for i in range(len(self.along_c)):
        if peak_i - min_points < i < peak_i + min_points:
            if abs(self.vacuum_locpot - self.locpot_along_c[i]) > conv_within:
                all_flat.append(False)
            else:
                all_flat.append(True)
    return all(all_flat)