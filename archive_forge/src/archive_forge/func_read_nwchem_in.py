import re
import numpy as np
from ase import Atoms
from ase.geometry import cellpar_to_cell
from .parser import _define_pattern
def read_nwchem_in(fobj, index=-1):
    text = ''.join(fobj.readlines())
    atomslist = []
    for match in _geom.findall(text):
        symbols = []
        positions = []
        for atom in _species.findall(match):
            atom = atom.split()
            symbols.append(atom[0])
            positions.append([float(x) for x in atom[1:]])
        positions = np.array(positions)
        atoms = Atoms(symbols)
        cell, pbc = _get_cell(text)
        pos = np.zeros_like(positions)
        for dim, ipbc in enumerate(pbc):
            if ipbc:
                pos += np.outer(positions[:, dim], cell[dim, :])
            else:
                pos[:, dim] = positions[:, dim]
        atoms.set_cell(cell)
        atoms.pbc = pbc
        atoms.set_positions(pos)
        atomslist.append(atoms)
    return atomslist[index]