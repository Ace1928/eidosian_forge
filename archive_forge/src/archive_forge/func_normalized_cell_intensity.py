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
def normalized_cell_intensity(self, structure: Structure, bragg_angles: dict[tuple[int, int, int], float]) -> dict[tuple[int, int, int], float]:
    """
        Normalizes the cell_intensity dict to 1, for use in plotting.

        Args:
            structure (Structure): The input structure.
            bragg_angles (dict of 3-tuple to float): The Bragg angles for each hkl plane.

        Returns:
            dict of hkl plane to normalized cell intensity
        """
    normalized_cell_intensity = {}
    cell_intensity = self.cell_intensity(structure, bragg_angles)
    max_intensity = max(cell_intensity.values())
    norm_factor = 1 / max_intensity
    for plane in cell_intensity:
        normalized_cell_intensity[plane] = cell_intensity[plane] * norm_factor
    return normalized_cell_intensity