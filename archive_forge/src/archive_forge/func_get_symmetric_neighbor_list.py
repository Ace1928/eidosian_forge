from __future__ import annotations
import collections
import contextlib
import functools
import inspect
import io
import itertools
import json
import math
import os
import random
import re
import sys
import warnings
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from inspect import isclass
from io import StringIO
from typing import TYPE_CHECKING, Any, Callable, Literal, SupportsIndex, cast, get_args
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from numpy import cross, eye
from numpy.linalg import norm
from ruamel.yaml import YAML
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import expm, polar
from scipy.spatial.distance import squareform
from tabulate import tabulate
from pymatgen.core.bonds import CovalentBond, get_bond_length
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice, get_points_in_spheres
from pymatgen.core.operations import SymmOp
from pymatgen.core.periodic_table import DummySpecies, Element, Species, get_el_sp
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.units import Length, Mass
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.maggroups import MagneticSpaceGroup
from pymatgen.util.coord import all_distances, get_angle, lattice_points_in_supercell
def get_symmetric_neighbor_list(self, r: float, sg: str, unique: bool=False, numerical_tol: float=1e-08, exclude_self: bool=True) -> tuple[np.ndarray, ...]:
    """Similar to 'get_neighbor_list' with sites=None, but the neighbors are
        grouped by symmetry. The returned values are a tuple of numpy arrays
        (center_indices, points_indices, offset_vectors, distances, symmetry_indices).
        Atom `center_indices[i]` has neighbor atom `points_indices[i]` that is translated
        by `offset_vectors[i]` lattice vectors, and the distance is `distances[i]`.
        Symmetry_idx groups the bonds that are related by a symmetry of the provided space
        group and symmetry_op is the operation that relates the first bond of the same
        symmetry_idx to the respective atom. The first bond maps onto itself via the
        Identity. The output is sorted w.r.t. to symmetry_indices. If unique is True only
        one of the two bonds connecting two points is given. Out of the two, the bond that
        does not reverse the sites is chosen.

        Args:
            r (float): Radius of sphere
            sg (str/int): The spacegroup the symmetry operations of which will be
                used to classify the neighbors. If a string, it will be interpreted
                as one of the notations supported by
                pymatgen.symmetry.groups.Spacegroup. E.g., "R-3c" or "Fm-3m".
                If an int, it will be interpreted as an international number.
                If None, 'get_space_group_info' will be used to determine the
                space group, default to None.
            unique (bool): Whether a bond is given for both, or only a single
                direction is given. The default is False.
            numerical_tol (float): This is a numerical tolerance for distances.
                Sites which are < numerical_tol are determined to be coincident
                with the site. Sites which are r + numerical_tol away is deemed
                to be within r from the site. The default of 1e-8 should be
                ok in most instances.
            exclude_self (bool): whether to exclude atom neighboring with itself within
                numerical tolerance distance, default to True

        Returns:
            tuple: (center_indices, points_indices, offset_vectors, distances,
                symmetry_indices, symmetry_ops)
        """
    from pymatgen.symmetry.groups import SpaceGroup
    if sg is None:
        ops = SpaceGroup(self.get_space_group_info()[0]).symmetry_ops
    else:
        try:
            sgp = SpaceGroup.from_int_number(int(sg))
        except ValueError:
            sgp = SpaceGroup(sg)
        ops = sgp.symmetry_ops
    lattice = self.lattice
    if not sgp.is_compatible(lattice):
        raise ValueError(f'Supplied lattice with parameters {lattice.parameters} is incompatible with supplied spacegroup {sgp.symbol}!')
    bonds = self.get_neighbor_list(r)
    if unique:
        redundant = []
        for idx, (i, j, R, d) in enumerate(zip(*bonds)):
            if idx in redundant:
                continue
            for jdx, (i2, j2, R2, d2) in enumerate(zip(*bonds)):
                bool1 = i == j2
                bool2 = j == i2
                bool3 = (-R2 == R).all()
                bool4 = np.allclose(d, d2, atol=numerical_tol)
                if bool1 and bool2 and bool3 and bool4:
                    redundant.append(jdx)
        m = ~np.in1d(np.arange(len(bonds[0])), redundant)
        idcs_dist = np.argsort(bonds[3][m])
        bonds = (bonds[0][m][idcs_dist], bonds[1][m][idcs_dist], bonds[2][m][idcs_dist], bonds[3][m][idcs_dist])
    n_bonds = len(bonds[0])
    symmetry_indices = np.empty(n_bonds)
    symmetry_indices[:] = np.nan
    symmetry_ops = np.empty(len(symmetry_indices), dtype=object)
    symmetry_identity = SymmOp.from_rotation_and_translation(np.eye(3), np.zeros(3))
    symmetry_index = 0
    for idx in range(n_bonds):
        if np.isnan(symmetry_indices[idx]):
            symmetry_indices[idx] = symmetry_index
            symmetry_ops[idx] = symmetry_identity
            for jdx in np.arange(n_bonds)[np.isnan(symmetry_indices)]:
                equal_distance = np.allclose(bonds[3][idx], bonds[3][jdx], atol=numerical_tol)
                if equal_distance:
                    from_a = self[bonds[0][idx]].frac_coords
                    to_a = self[bonds[1][idx]].frac_coords
                    r_a = bonds[2][idx]
                    from_b = self[bonds[0][jdx]].frac_coords
                    to_b = self[bonds[1][jdx]].frac_coords
                    r_b = bonds[2][jdx]
                    for op in ops:
                        are_related, is_reversed = op.are_symmetrically_related_vectors(from_a, to_a, r_a, from_b, to_b, r_b)
                        if are_related and (not is_reversed):
                            symmetry_indices[jdx] = symmetry_index
                            symmetry_ops[jdx] = op
                        elif are_related and is_reversed:
                            symmetry_indices[jdx] = symmetry_index
                            symmetry_ops[jdx] = op
                            bonds[0][jdx], bonds[1][jdx] = (bonds[1][jdx], bonds[0][jdx])
                            bonds[2][jdx] = -bonds[2][jdx]
            symmetry_index += 1
    idcs_symid = np.argsort(symmetry_indices)
    bonds = (bonds[0][idcs_symid], bonds[1][idcs_symid], bonds[2][idcs_symid], bonds[3][idcs_symid])
    symmetry_indices = symmetry_indices[idcs_symid]
    symmetry_ops = symmetry_ops[idcs_symid]
    idcs_symop = np.arange(n_bonds)
    identity_idcs = np.where(symmetry_ops == symmetry_identity)[0]
    for symmetry_idx in np.unique(symmetry_indices):
        first_idx = np.argmax(symmetry_indices == symmetry_idx)
        for second_idx in identity_idcs:
            if symmetry_indices[second_idx] == symmetry_idx:
                idcs_symop[first_idx], idcs_symop[second_idx] = (idcs_symop[second_idx], idcs_symop[first_idx])
    return (bonds[0][idcs_symop], bonds[1][idcs_symop], bonds[2][idcs_symop], bonds[3][idcs_symop], symmetry_indices[idcs_symop], symmetry_ops[idcs_symop])