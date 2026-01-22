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
def get_interplanar_angle(structure: Structure, p1: tuple[int, int, int], p2: tuple[int, int, int]) -> float:
    """
        Returns the interplanar angle (in degrees) between the normal of two crystal planes.
        Formulas from International Tables for Crystallography Volume C pp. 2-9.

        Args:
            structure (Structure): The input structure.
            p1 (3-tuple): plane 1
            p2 (3-tuple): plane 2

        Returns:
            float
        """
    a, b, c = (structure.lattice.a, structure.lattice.b, structure.lattice.c)
    alpha, beta, gamma = (np.deg2rad(structure.lattice.alpha), np.deg2rad(structure.lattice.beta), np.deg2rad(structure.lattice.gamma))
    vol = structure.volume
    a_star = b * c * np.sin(alpha) / vol
    b_star = a * c * np.sin(beta) / vol
    c_star = a * b * np.sin(gamma) / vol
    cos_alpha_star = (np.cos(beta) * np.cos(gamma) - np.cos(alpha)) / (np.sin(beta) * np.sin(gamma))
    cos_beta_star = (np.cos(alpha) * np.cos(gamma) - np.cos(beta)) / (np.sin(alpha) * np.sin(gamma))
    cos_gamma_star = (np.cos(alpha) * np.cos(beta) - np.cos(gamma)) / (np.sin(alpha) * np.sin(beta))
    r1_norm = np.sqrt(p1[0] ** 2 * a_star ** 2 + p1[1] ** 2 * b_star ** 2 + p1[2] ** 2 * c_star ** 2 + 2 * p1[0] * p1[1] * a_star * b_star * cos_gamma_star + 2 * p1[0] * p1[2] * a_star * c_star * cos_beta_star + 2 * p1[1] * p1[2] * b_star * c_star * cos_gamma_star)
    r2_norm = np.sqrt(p2[0] ** 2 * a_star ** 2 + p2[1] ** 2 * b_star ** 2 + p2[2] ** 2 * c_star ** 2 + 2 * p2[0] * p2[1] * a_star * b_star * cos_gamma_star + 2 * p2[0] * p2[2] * a_star * c_star * cos_beta_star + 2 * p2[1] * p2[2] * b_star * c_star * cos_gamma_star)
    r1_dot_r2 = p1[0] * p2[0] * a_star ** 2 + p1[1] * p2[1] * b_star ** 2 + p1[2] * p2[2] * c_star ** 2 + (p1[0] * p2[1] + p2[0] * p1[1]) * a_star * b_star * cos_gamma_star + (p1[0] * p2[2] + p2[0] * p1[1]) * a_star * c_star * cos_beta_star + (p1[1] * p2[2] + p2[1] * p1[2]) * b_star * c_star * cos_alpha_star
    phi = np.arccos(r1_dot_r2 / (r1_norm * r2_norm))
    return np.rad2deg(phi)