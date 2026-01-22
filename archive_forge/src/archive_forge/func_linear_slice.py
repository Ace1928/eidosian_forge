from __future__ import annotations
import itertools
import json
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from scipy.interpolate import RegularGridInterpolator
from pymatgen.core import Element, Site, Structure
from pymatgen.core.units import ang_to_bohr, bohr_to_angstrom
from pymatgen.electronic_structure.core import Spin
def linear_slice(self, p1, p2, n=100):
    """
        Get a linear slice of the volumetric data with n data points from
        point p1 to point p2, in the form of a list.

        Args:
            p1 (list): 3-element list containing fractional coordinates of the first point.
            p2 (list): 3-element list containing fractional coordinates of the second point.
            n (int): Number of data points to collect, defaults to 100.

        Returns:
            List of n data points (mostly interpolated) representing a linear slice of the
            data from point p1 to point p2.
        """
    assert type(p1) in [list, np.ndarray]
    assert type(p2) in [list, np.ndarray]
    assert len(p1) == 3
    assert len(p2) == 3
    xpts = np.linspace(p1[0], p2[0], num=n)
    ypts = np.linspace(p1[1], p2[1], num=n)
    zpts = np.linspace(p1[2], p2[2], num=n)
    return [self.value_at(xpts[i], ypts[i], zpts[i]) for i in range(n)]