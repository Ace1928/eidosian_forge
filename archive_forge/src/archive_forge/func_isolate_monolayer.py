import itertools
import collections
import numpy as np
from ase import Atoms
from ase.geometry.cell import complete_cell
from ase.geometry.dimensionality import analyze_dimensionality
from ase.geometry.dimensionality import rank_determination
from ase.geometry.dimensionality.bond_generator import next_bond
from ase.geometry.dimensionality.interval_analysis import merge_intervals
def isolate_monolayer(atoms, components, k, v):
    a, b, basis = construct_inplane_basis(atoms, k, v)
    c = np.cross(a, b)
    c /= np.linalg.norm(c)
    init_cell = np.dot(np.array([a, b, c]), basis.T)
    positions, numbers = build_supercomponent(atoms, components, k, v)
    scaled = np.linalg.solve(init_cell.T, np.dot(positions, basis.T).T).T
    scaled[:, :2] %= 1.0
    scaled[:, 2] -= np.mean(scaled, axis=0)[2]
    pos = np.dot(scaled, init_cell)
    zmax = np.max(np.abs(pos[:, 2]))
    cell = np.copy(init_cell)
    cell[2] *= 4 * zmax
    return Atoms(numbers=numbers, positions=pos, cell=cell, pbc=[1, 1, 0])