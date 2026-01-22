import itertools
import numpy as np
from ase.lattice import bravais_lattices, UnconventionalLattice, bravais_names
from ase.cell import Cell
def generate_niggli_op_table(lattices=None, length_grid=None, angle_grid=None):
    if length_grid is None:
        length_grid = np.logspace(-0.5, 1.5, 50).round(3)
    if angle_grid is None:
        angle_grid = np.linspace(10, 179, 50).round()
    all_niggli_ops_and_counts = find_all_niggli_ops(length_grid, angle_grid, lattices=lattices)
    niggli_op_table = {}
    for latname, ops in all_niggli_ops_and_counts.items():
        ops = [op for op in ops if np.abs(op).max() < 2]
        niggli_op_table[latname] = ops
    import pprint
    print(pprint.pformat(niggli_op_table))
    return niggli_op_table