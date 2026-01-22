import itertools
import numpy as np
from ase.lattice import bravais_lattices, UnconventionalLattice, bravais_names
from ase.cell import Cell
def find_niggli_ops(latcls, length_grid, angle_grid):
    niggli_ops = {}
    for lat in lattice_loop(latcls, length_grid, angle_grid):
        cell = lat.tocell()
        try:
            rcell, op = cell.niggli_reduce()
        except RuntimeError:
            print('Niggli reduce did not converge')
            continue
        assert op.dtype == int
        op_key = tuple(op.ravel())
        if op_key in niggli_ops:
            niggli_ops[op_key] += 1
        else:
            niggli_ops[op_key] = 1
        rcell_test = Cell(op.T @ cell)
        rcellpar_test = rcell_test.cellpar()
        rcellpar = rcell.cellpar()
        err = np.abs(rcellpar_test - rcellpar).max()
        assert err < 1e-07, err
    return niggli_ops