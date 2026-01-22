from __future__ import annotations
import logging
import warnings
from fractions import Fraction
from functools import reduce
from itertools import chain, combinations, product
from math import cos, floor, gcd
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.fractions import lcm
from numpy.testing import assert_allclose
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.lattice import Lattice
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@staticmethod
def get_rotation_angle_from_sigma(sigma, r_axis, lat_type='C', ratio=None):
    """
        Find all possible rotation angle for the given sigma value.

        Args:
            sigma (int): sigma value provided
            r_axis (list of 3 integers, e.g. u, v, w or 4 integers, e.g. u, v, t, w for hex/rho system only): the
                rotation axis of the grain boundary.
            lat_type (str): one character to specify the lattice type. Defaults to 'c' for cubic.
                'c' or 'C': cubic system
                't' or 'T': tetragonal system
                'o' or 'O': orthorhombic system
                'h' or 'H': hexagonal system
                'r' or 'R': rhombohedral system
            ratio (list of integers): lattice axial ratio.
                For cubic system, ratio is not needed.
                For tetragonal system, ratio = [mu, mv], list of two integers,
                that is, mu/mv = c2/a2. If it is irrational, set it to none.
                For orthorhombic system, ratio = [mu, lam, mv], list of 3 integers,
                that is, mu:lam:mv = c2:b2:a2. If irrational for one axis, set it to None.
                e.g. mu:lam:mv = c2,None,a2, means b2 is irrational.
                For rhombohedral system, ratio = [mu, mv], list of two integers,
                that is, mu/mv is the ratio of (1+2*cos(alpha)/cos(alpha).
                If irrational, set it to None.
                For hexagonal system, ratio = [mu, mv], list of two integers,
                that is, mu/mv = c2/a2. If it is irrational, set it to none.

        Returns:
            rotation_angles corresponding to the provided sigma value.
            If the sigma value is not correct, return the rotation angle corresponding
            to the correct possible sigma value right smaller than the wrong sigma value provided.
        """
    if lat_type.lower() == 'c':
        logger.info('Make sure this is for cubic system')
        sigma_dict = GrainBoundaryGenerator.enum_sigma_cubic(cutoff=sigma, r_axis=r_axis)
    elif lat_type.lower() == 't':
        logger.info('Make sure this is for tetragonal system')
        if ratio is None:
            logger.info('Make sure this is for irrational c2/a2 ratio')
        elif len(ratio) != 2:
            raise RuntimeError('Tetragonal system needs correct c2/a2 ratio')
        sigma_dict = GrainBoundaryGenerator.enum_sigma_tet(cutoff=sigma, r_axis=r_axis, c2_a2_ratio=ratio)
    elif lat_type.lower() == 'o':
        logger.info('Make sure this is for orthorhombic system')
        if len(ratio) != 3:
            raise RuntimeError('Orthorhombic system needs correct c2:b2:a2 ratio')
        sigma_dict = GrainBoundaryGenerator.enum_sigma_ort(cutoff=sigma, r_axis=r_axis, c2_b2_a2_ratio=ratio)
    elif lat_type.lower() == 'h':
        logger.info('Make sure this is for hexagonal system')
        if ratio is None:
            logger.info('Make sure this is for irrational c2/a2 ratio')
        elif len(ratio) != 2:
            raise RuntimeError('Hexagonal system needs correct c2/a2 ratio')
        sigma_dict = GrainBoundaryGenerator.enum_sigma_hex(cutoff=sigma, r_axis=r_axis, c2_a2_ratio=ratio)
    elif lat_type.lower() == 'r':
        logger.info('Make sure this is for rhombohedral system')
        if ratio is None:
            logger.info('Make sure this is for irrational (1+2*cos(alpha)/cos(alpha) ratio')
        elif len(ratio) != 2:
            raise RuntimeError('Rhombohedral system needs correct (1+2*cos(alpha)/cos(alpha) ratio')
        sigma_dict = GrainBoundaryGenerator.enum_sigma_rho(cutoff=sigma, r_axis=r_axis, ratio_alpha=ratio)
    else:
        raise RuntimeError('Lattice type not implemented')
    sigmas = list(sigma_dict)
    if not sigmas:
        raise RuntimeError('This is a wrong sigma value, and no sigma exists smaller than this value.')
    if sigma in sigmas:
        rotation_angles = sigma_dict[sigma]
    else:
        sigmas.sort()
        warnings.warn('This is not the possible sigma value according to the rotation axis!The nearest neighbor sigma and its corresponding angle are returned')
        rotation_angles = sigma_dict[sigmas[-1]]
    rotation_angles.sort()
    return rotation_angles