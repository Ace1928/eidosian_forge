from __future__ import annotations
import json
import os
from collections import namedtuple
from fractions import Fraction
from typing import TYPE_CHECKING, cast
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.constants as sc
from pymatgen.analysis.diffraction.core import AbstractDiffractionPatternCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.string import latexify_spacegroup, unicodeify_spacegroup
@staticmethod
def generate_points(coord_left: int=-10, coord_right: int=10) -> np.ndarray:
    """
        Generates a bunch of 3D points that span a cube.

        Args:
            coord_left (int): The minimum coordinate value.
            coord_right (int): The maximum coordinate value.

        Returns:
            np.array: 2d array
        """
    points = [0, 0, 0]
    coord_values = np.arange(coord_left, coord_right + 1)
    points[0], points[1], points[2] = np.meshgrid(coord_values, coord_values, coord_values)
    points_matrix = (np.ravel(points[i]) for i in range(3))
    return np.vstack(list(points_matrix)).transpose()