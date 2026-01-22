from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
def identify_lattice(cell, eps=0.0002, *, pbc=True):
    """Find Bravais lattice representing this cell.

    Returns Bravais lattice object representing the cell along with
    an operation that, applied to the cell, yields the same lengths
    and angles as the Bravais lattice object."""
    pbc = cell.any(1) & pbc2pbc(pbc)
    npbc = sum(pbc)
    if npbc == 1:
        i = np.argmax(pbc)
        a = cell[i, i]
        if a < 0 or cell[i, [i - 1, i - 2]].any():
            raise ValueError('Not a 1-d cell ASE can handle: {cell}.'.format(cell=cell))
        if i == 0:
            op = np.eye(3)
        elif i == 1:
            op = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        else:
            op = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        return (LINE(a), op)
    if npbc == 2:
        lat, op = get_2d_bravais_lattice(cell, eps, pbc=pbc)
        return (lat, op)
    if npbc != 3:
        raise ValueError('System must be periodic either along all three axes, along two first axes or, along the third axis.  Got pbc={}'.format(pbc))
    from ase.geometry.bravais_type_engine import niggli_op_table
    if cell.rank < 3:
        raise ValueError('Expected 3 linearly independent cell vectors')
    rcell, reduction_op = cell.niggli_reduce(eps=eps)
    memory = {}
    for latname in LatticeChecker.check_order:
        matching_lattices = []
        for op_key in niggli_op_table[latname]:
            checker_and_op = memory.get(op_key)
            if checker_and_op is None:
                normalization_op = np.array(op_key).reshape(3, 3)
                candidate = Cell(np.linalg.inv(normalization_op.T) @ rcell)
                checker = LatticeChecker(candidate, eps=eps)
                memory[op_key] = (checker, normalization_op)
            else:
                checker, normalization_op = checker_and_op
            lat = checker.query(latname)
            if lat is not None:
                op = normalization_op @ np.linalg.inv(reduction_op)
                matching_lattices.append((lat, op))
        best = None
        best_defect = np.inf
        for lat, op in matching_lattices:
            cell = lat.tocell()
            lengths = cell.lengths()
            defect = np.prod(lengths) / cell.volume
            if defect < best_defect:
                best = (lat, op)
                best_defect = defect
        if best is not None:
            return best