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
def electron_scattering_factors(self, structure: Structure, bragg_angles: dict[tuple[int, int, int], float]) -> dict[str, dict[tuple[int, int, int], float]]:
    """
        Calculates atomic scattering factors for electrons using the Mott-Bethe formula (1st order Born approximation).

        Args:
            structure (Structure): The input structure.
            bragg_angles (dict of 3-tuple to float): The Bragg angles for each hkl plane.

        Returns:
            dict from atomic symbol to another dict of hkl plane to factor (in angstroms)
        """
    electron_scattering_factors = {}
    x_ray_factors = self.x_ray_factors(structure, bragg_angles)
    s2 = self.get_s2(bragg_angles)
    atoms = structure.elements
    prefactor = 0.023934
    scattering_factors_for_atom = {}
    for atom in atoms:
        for plane in bragg_angles:
            scattering_factor_curr = prefactor * (atom.Z - x_ray_factors[atom.symbol][plane]) / s2[plane]
            scattering_factors_for_atom[plane] = scattering_factor_curr
        electron_scattering_factors[atom.symbol] = scattering_factors_for_atom
        scattering_factors_for_atom = {}
    return electron_scattering_factors