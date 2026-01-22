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
def cell_scattering_factors(self, structure: Structure, bragg_angles: dict[tuple[int, int, int], float]) -> dict[tuple[int, int, int], int]:
    """
        Calculates the scattering factor for the whole cell.

        Args:
            structure (Structure): The input structure.
            bragg_angles (dict of 3-tuple to float): The Bragg angles for each hkl plane.

        Returns:
            dict of hkl plane (3-tuple) to scattering factor (in angstroms).
        """
    cell_scattering_factors = {}
    electron_scattering_factors = self.electron_scattering_factors(structure, bragg_angles)
    scattering_factor_curr = 0
    for plane in bragg_angles:
        for site in structure:
            for sp in site.species:
                g_dot_r = np.dot(np.array(plane), np.transpose(site.frac_coords))
                scattering_factor_curr += electron_scattering_factors[sp.symbol][plane] * np.exp(2j * np.pi * g_dot_r)
        cell_scattering_factors[plane] = scattering_factor_curr
        scattering_factor_curr = 0
    return cell_scattering_factors