import os
import re
from subprocess import call, TimeoutExpired
from copy import deepcopy
import numpy as np
from ase import Atoms
from ase.utils import workdir
from ase.units import Hartree, Bohr, Debye
from ase.calculators.singlepoint import SinglePointCalculator
def read_gamess_us_punch(fd):
    atoms = None
    energy = None
    forces = None
    dipole = None
    for line in fd:
        if line.strip() == '$DATA':
            symbols = []
            pos = []
            while line.strip() != '$END':
                line = fd.readline()
                atom = _atom_re.match(line)
                if atom is None:
                    continue
                symbols.append(atom.group(1).capitalize())
                pos.append(list(map(float, atom.group(3, 4, 5))))
            atoms = Atoms(symbols, np.array(pos))
        elif line.startswith('E('):
            energy = float(line.split()[1][:-1]) * Hartree
        elif line.strip().startswith('DIPOLE'):
            dipole = np.array(list(map(float, line.split()[1:]))) * Debye
        elif line.strip() == '$GRAD':
            energy = float(fd.readline().split()[1]) * Hartree
            grad = []
            while line.strip() != '$END':
                line = fd.readline()
                atom = _atom_re.match(line)
                if atom is None:
                    continue
                grad.append(list(map(float, atom.group(3, 4, 5))))
            forces = -np.array(grad) * Hartree / Bohr
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces, dipole=dipole)
    return atoms