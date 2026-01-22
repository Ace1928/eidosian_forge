from __future__ import annotations
import json
import math
import os
import warnings
from bisect import bisect_left
from collections import defaultdict, namedtuple
from copy import deepcopy
from functools import lru_cache
from math import acos, asin, atan2, cos, exp, fabs, pi, pow, sin, sqrt
from typing import TYPE_CHECKING, Any, Literal, get_args
import numpy as np
from monty.dev import deprecated, requires
from monty.serialization import loadfn
from ruamel.yaml import YAML
from scipy.spatial import Voronoi
from pymatgen.analysis.bond_valence import BV_PARAMS, BVAnalyzer
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core import Element, IStructure, PeriodicNeighbor, PeriodicSite, Site, Species, Structure
def get_q4(self, thetas=None, phis=None):
    """
        Calculates the value of the bond orientational order parameter of
        weight l=4. If the function is called with non-empty lists of
        polar and azimuthal angles the corresponding trigonometric terms
        are computed afresh. Otherwise, it is expected that the
        compute_trigonometric_terms function has been just called.

        Args:
            thetas ([float]): polar angles of all neighbors in radians.
            phis ([float]): azimuth angles of all neighbors in radians.

        Returns:
            float: bond orientational order parameter of weight l=4
                corresponding to the input angles thetas and phis.
        """
    if thetas is not None and phis is not None:
        self.compute_trigonometric_terms(thetas, phis)
    n_nn = len(self._pow_sin_t[1])
    n_nn_range = range(n_nn)
    i16_3 = 3 / 16.0
    i8_3 = 3 / 8.0
    sqrt_35_pi = sqrt(35 / pi)
    sqrt_35_2pi = sqrt(35 / (2 * pi))
    sqrt_5_pi = sqrt(5 / pi)
    sqrt_5_2pi = sqrt(5 / (2 * pi))
    sqrt_1_pi = sqrt(1 / pi)
    pre_y_4_4 = [i16_3 * sqrt_35_2pi * val for val in self._pow_sin_t[4]]
    pre_y_4_3 = [i8_3 * sqrt_35_pi * val[0] * val[1] for val in zip(self._pow_sin_t[3], self._pow_cos_t[1])]
    pre_y_4_2 = [i8_3 * sqrt_5_2pi * val[0] * (7 * val[1] - 1.0) for val in zip(self._pow_sin_t[2], self._pow_cos_t[2])]
    pre_y_4_1 = [i8_3 * sqrt_5_pi * val[0] * (7 * val[1] - 3 * val[2]) for val in zip(self._pow_sin_t[1], self._pow_cos_t[3], self._pow_cos_t[1])]
    acc = 0.0
    real = imag = 0.0
    for idx in n_nn_range:
        real += pre_y_4_4[idx] * self._cos_n_p[4][idx]
        imag -= pre_y_4_4[idx] * self._sin_n_p[4][idx]
    acc += real * real + imag * imag
    real = imag = 0.0
    for idx in n_nn_range:
        real += pre_y_4_3[idx] * self._cos_n_p[3][idx]
        imag -= pre_y_4_3[idx] * self._sin_n_p[3][idx]
    acc += real * real + imag * imag
    real = imag = 0.0
    for idx in n_nn_range:
        real += pre_y_4_2[idx] * self._cos_n_p[2][idx]
        imag -= pre_y_4_2[idx] * self._sin_n_p[2][idx]
    acc += real * real + imag * imag
    real = imag = 0.0
    for idx in n_nn_range:
        real += pre_y_4_1[idx] * self._cos_n_p[1][idx]
        imag -= pre_y_4_1[idx] * self._sin_n_p[1][idx]
    acc += real * real + imag * imag
    real = imag = 0.0
    for idx in n_nn_range:
        real += i16_3 * sqrt_1_pi * (35 * self._pow_cos_t[4][idx] - 30 * self._pow_cos_t[2][idx] + 3.0)
    acc += real * real
    real = imag = 0.0
    for idx in n_nn_range:
        real -= pre_y_4_1[idx] * self._cos_n_p[1][idx]
        imag -= pre_y_4_1[idx] * self._sin_n_p[1][idx]
    acc += real * real + imag * imag
    real = imag = 0.0
    for idx in n_nn_range:
        real += pre_y_4_2[idx] * self._cos_n_p[2][idx]
        imag += pre_y_4_2[idx] * self._sin_n_p[2][idx]
    acc += real * real + imag * imag
    real = imag = 0.0
    for idx in n_nn_range:
        real -= pre_y_4_3[idx] * self._cos_n_p[3][idx]
        imag -= pre_y_4_3[idx] * self._sin_n_p[3][idx]
    acc += real * real + imag * imag
    real = imag = 0.0
    for idx in n_nn_range:
        real += pre_y_4_4[idx] * self._cos_n_p[4][idx]
        imag += pre_y_4_4[idx] * self._sin_n_p[4][idx]
    acc += real * real + imag * imag
    return sqrt(4 * pi * acc / (9 * float(n_nn * n_nn)))