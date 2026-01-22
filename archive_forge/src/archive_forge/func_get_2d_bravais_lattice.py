from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
def get_2d_bravais_lattice(origcell, eps=0.0002, *, pbc=True):
    pbc = origcell.any(1) & pbc2pbc(pbc)
    if list(pbc) != [1, 1, 0]:
        raise UnsupportedLattice('Can only get 2D Bravais lattice of cell with pbc==[1, 1, 0]; but we have {}'.format(pbc))
    nonperiodic = pbc.argmin()
    ops = [np.eye(3)]
    for i in range(-1, 1):
        for j in range(-1, 1):
            op = [[1, j], [i, 1]]
            if np.abs(np.linalg.det(op)) > 1e-05:
                op = np.insert(op, nonperiodic, [0, 0], 0)
                op = np.insert(op, nonperiodic, ~pbc, 1)
                ops.append(np.array(op))

    def allclose(a, b):
        return np.allclose(a, b, atol=eps)
    symrank = 0
    for op in ops:
        cell = Cell(op.dot(origcell))
        cellpar = cell.cellpar()
        angles = cellpar[3:]
        gamma = angles[~pbc][0]
        a, b = cellpar[:3][pbc]
        anglesm90 = np.abs(angles - 90)
        if np.sum(anglesm90 > eps) > 1:
            continue
        all_lengths_equal = abs(a - b) < eps
        if all_lengths_equal:
            if allclose(gamma, 90):
                lat = SQR(a)
                rank = 5
            elif allclose(gamma, 120):
                lat = HEX2D(a)
                rank = 4
            else:
                lat = CRECT(a, gamma)
                rank = 3
        elif allclose(gamma, 90):
            lat = RECT(a, b)
            rank = 2
        else:
            lat = OBL(a, b, gamma)
            rank = 1
        op = lat.get_transformation(origcell, eps=eps)
        if not allclose(np.dot(op, lat.tocell())[pbc][:, pbc], origcell.array[pbc][:, pbc]):
            msg = 'Cannot recognize cell at all somehow! {}, {}, {}'.format(a, b, gamma)
            raise RuntimeError(msg)
        if rank > symrank:
            finalop = op
            symrank = rank
            finallat = lat
    return (finallat, finalop.T)