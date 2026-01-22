from __future__ import annotations
import copy
import itertools
import logging
import math
import warnings
from collections import defaultdict
from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from math import cos, sin
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
import scipy.cluster
import spglib
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Molecule, PeriodicSite, Structure
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list, pbc_diff
from pymatgen.util.due import Doi, due
def get_ir_reciprocal_mesh_map(self, mesh=(10, 10, 10), is_shift=(0, 0, 0)):
    """Same as 'get_ir_reciprocal_mesh' but the full grid together with the mapping
        that maps a reducible to an irreducible kpoint is returned.

        Args:
            mesh (3x1 array): The number of kpoint for the mesh needed in
                each direction
            is_shift (3x1 array): Whether to shift the kpoint grid. (1, 1,
            1) means all points are shifted by 0.5, 0.5, 0.5.

        Returns:
            A tuple containing two numpy.ndarray. The first is the mesh in
            fractional coordinates and the second is an array of integers
            that maps all the reducible kpoints from to irreducible ones.
        """
    shift = np.array([1 if i else 0 for i in is_shift])
    mapping, grid = spglib.get_ir_reciprocal_mesh(np.array(mesh), self._cell, is_shift=shift, symprec=self._symprec)
    grid_fractional_coords = (grid + shift * (0.5, 0.5, 0.5)) / mesh
    return (grid_fractional_coords, mapping)