from pathlib import Path
from re import compile
import numpy as np
from ase import Atoms
from ase.utils import reader
from ase.units import Bohr
def read_siesta_xv(fd):
    vectors = []
    for i in range(3):
        data = next(fd).split()
        vectors.append([float(data[j]) * Bohr for j in range(3)])
    natoms = int(next(fd).split()[0])
    speciesnumber, atomnumbers, xyz, V = ([], [], [], [])
    for line in fd:
        if len(line) > 5:
            data = line.split()
            speciesnumber.append(int(data[0]))
            atomnumbers.append(int(data[1]))
            xyz.append([float(data[2 + j]) * Bohr for j in range(3)])
            V.append([float(data[5 + j]) * Bohr for j in range(3)])
    vectors = np.array(vectors)
    atomnumbers = np.array(atomnumbers)
    xyz = np.array(xyz)
    atoms = Atoms(numbers=atomnumbers, positions=xyz, cell=vectors, pbc=True)
    assert natoms == len(atoms)
    return atoms