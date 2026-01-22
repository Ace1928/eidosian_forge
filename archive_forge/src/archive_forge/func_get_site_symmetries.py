from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_site_symmetries(struct: Structure, precision: float=0.1) -> list[list[SymmOp]]:
    """Get all the point group operations centered on each atomic site
    in the form [[point operations of site index 1]...[[point operations of site index N]]].

    Args:
        struct: Pymatgen structure
        precision (float): tolerance to find symmetry operations

    Returns:
        list of lists of point operations for each atomic site
    """
    point_ops: list[list[SymmOp]] = []
    for idx1 in range(len(struct)):
        temp_struct = struct.copy()
        point_ops.append([])
        for idx2, site2 in enumerate(struct):
            temp_struct.replace(idx2, site2.specie, temp_struct.frac_coords[idx2] - struct.frac_coords[idx1])
        sga_struct = SpacegroupAnalyzer(temp_struct, symprec=precision)
        ops = sga_struct.get_symmetry_operations(cartesian=True)
        point_ops[idx1] = [op for op in ops if list(op.translation_vector) == [0, 0, 0]]
    return point_ops