import os
from os.path import join
import re
from glob import glob
import warnings
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.units import Hartree, Bohr, fs
from ase.calculators.calculator import Parameters
def read_abinit_in(fd):
    """Import ABINIT input file.

    Reads cell, atom positions, etc. from abinit input file
    """
    tokens = []
    for line in fd:
        meat = line.split('#', 1)[0]
        tokens += meat.lower().split()
    index = tokens.index('acell')
    unit = 1.0
    if tokens[index + 4].lower()[:3] != 'ang':
        unit = Bohr
    acell = [unit * float(tokens[index + 1]), unit * float(tokens[index + 2]), unit * float(tokens[index + 3])]
    index = tokens.index('natom')
    natom = int(tokens[index + 1])
    index = tokens.index('ntypat')
    ntypat = int(tokens[index + 1])
    index = tokens.index('typat')
    typat = []
    while len(typat) < natom:
        token = tokens[index + 1]
        if '*' in token:
            nrepeat, typenum = token.split('*')
            typat += [int(typenum)] * int(nrepeat)
        else:
            typat.append(int(token))
        index += 1
    assert natom == len(typat)
    index = tokens.index('znucl')
    znucl = []
    for i in range(ntypat):
        znucl.append(int(tokens[index + 1 + i]))
    index = tokens.index('rprim')
    rprim = []
    for i in range(3):
        rprim.append([acell[i] * float(tokens[index + 3 * i + 1]), acell[i] * float(tokens[index + 3 * i + 2]), acell[i] * float(tokens[index + 3 * i + 3])])
    numbers = []
    for i in range(natom):
        ii = typat[i] - 1
        numbers.append(znucl[ii])
    if 'xred' in tokens:
        index = tokens.index('xred')
        xred = []
        for i in range(natom):
            xred.append([float(tokens[index + 3 * i + 1]), float(tokens[index + 3 * i + 2]), float(tokens[index + 3 * i + 3])])
        atoms = Atoms(cell=rprim, scaled_positions=xred, numbers=numbers, pbc=True)
    else:
        if 'xcart' in tokens:
            index = tokens.index('xcart')
            unit = Bohr
        elif 'xangst' in tokens:
            unit = 1.0
            index = tokens.index('xangst')
        else:
            raise IOError('No xred, xcart, or xangs keyword in abinit input file')
        xangs = []
        for i in range(natom):
            xangs.append([unit * float(tokens[index + 3 * i + 1]), unit * float(tokens[index + 3 * i + 2]), unit * float(tokens[index + 3 * i + 3])])
        atoms = Atoms(cell=rprim, positions=xangs, numbers=numbers, pbc=True)
    try:
        ii = tokens.index('nsppol')
    except ValueError:
        nsppol = None
    else:
        nsppol = int(tokens[ii + 1])
    if nsppol == 2:
        index = tokens.index('spinat')
        magmoms = [float(tokens[index + 3 * i + 3]) for i in range(natom)]
        atoms.set_initial_magnetic_moments(magmoms)
    assert len(atoms) == natom
    return atoms