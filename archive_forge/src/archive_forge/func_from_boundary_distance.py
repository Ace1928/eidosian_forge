from __future__ import annotations
import logging
from fractions import Fraction
from typing import TYPE_CHECKING
import numpy as np
from numpy import around
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.elasticity.strain import Deformation
from pymatgen.analysis.ewald import EwaldMinimizer, EwaldSummation
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, get_el_sp
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.site_transformations import PartialRemoveSitesTransformation
from pymatgen.transformations.transformation_abc import AbstractTransformation
@classmethod
def from_boundary_distance(cls, structure: Structure, min_boundary_dist: float=6, allow_rotation: bool=False, max_atoms: float=-1) -> Self:
    """Get a SupercellTransformation according to the desired minimum distance between periodic
        boundaries of the resulting supercell.

        Args:
            structure (Structure): Input structure.
            min_boundary_dist (float): Desired minimum distance between all periodic boundaries. Defaults to 6.
            allow_rotation (bool): Whether allowing lattice angles to change. Only useful when
                at least two of the three lattice vectors are required to expand. Defaults to False.
                If True, a SupercellTransformation satisfying min_boundary_dist but with smaller
                number of atoms than the SupercellTransformation with unchanged lattice angles
                can possibly be found. If such a SupercellTransformation cannot be found easily,
                the SupercellTransformation with unchanged lattice angles will be returned.
            max_atoms (int): Maximum number of atoms allowed in the supercell. Defaults to -1 for infinity.

        Returns:
            SupercellTransformation.
        """
    min_expand = np.int8(min_boundary_dist / np.array([structure.lattice.d_hkl(plane) for plane in np.eye(3)]))
    max_atoms = max_atoms if max_atoms > 0 else float('inf')
    if allow_rotation and sum(min_expand != 0) > 1:
        min1, min2, min3 = map(int, min_expand)
        scaling_matrix = [[min1 or 1, 1 if min1 and min2 else 0, 1 if min1 and min3 else 0], [-1 if min2 and min1 else 0, min2 or 1, 1 if min2 and min3 else 0], [-1 if min3 and min1 else 0, -1 if min3 and min2 else 0, min3 or 1]]
        struct_scaled = structure.make_supercell(scaling_matrix, in_place=False)
        min_expand_scaled = np.int8(min_boundary_dist / np.array([struct_scaled.lattice.d_hkl(plane) for plane in np.eye(3)]))
        if sum(min_expand_scaled != 0) == 0 and len(struct_scaled) <= max_atoms:
            return cls(scaling_matrix)
    scaling_matrix = np.eye(3) + np.diag(min_expand)
    struct_scaled = structure.make_supercell(scaling_matrix, in_place=False)
    if len(struct_scaled) <= max_atoms:
        return cls(scaling_matrix)
    msg = f'max_atoms={max_atoms!r} exceeded while trying to solve for supercell. You can try lowering min_boundary_dist={min_boundary_dist!r}'
    if not allow_rotation:
        msg += ' or set allow_rotation=True'
    raise RuntimeError(msg)