import numpy as np
from ase import Atoms
from ase.units import Bohr, Ry
from ase.utils import reader, writer
@reader
def read_struct(fd, ase=True):
    pip = fd.readlines()
    lattice = pip[1][0:3]
    nat = int(pip[1][27:30])
    cell = np.zeros(6)
    for i in range(6):
        cell[i] = float(pip[3][0 + i * 10:10 + i * 10])
    cell[0:3] = cell[0:3] * Bohr
    if lattice == 'P  ':
        lattice = 'P'
    elif lattice == 'H  ':
        lattice = 'P'
        cell[3:6] = [90.0, 90.0, 120.0]
    elif lattice == 'R  ':
        lattice = 'R'
    elif lattice == 'F  ':
        lattice = 'F'
    elif lattice == 'B  ':
        lattice = 'I'
    elif lattice == 'CXY':
        lattice = 'C'
    elif lattice == 'CXZ':
        lattice = 'B'
    elif lattice == 'CYZ':
        lattice = 'A'
    else:
        raise RuntimeError('TEST needed')
    pos = np.array([])
    atomtype = []
    rmt = []
    neq = np.zeros(nat)
    iline = 4
    indif = 0
    for iat in range(nat):
        indifini = indif
        if len(pos) == 0:
            pos = np.array([[float(pip[iline][12:22]), float(pip[iline][25:35]), float(pip[iline][38:48])]])
        else:
            pos = np.append(pos, np.array([[float(pip[iline][12:22]), float(pip[iline][25:35]), float(pip[iline][38:48])]]), axis=0)
        indif += 1
        iline += 1
        neq[iat] = int(pip[iline][15:17])
        iline += 1
        for ieq in range(1, int(neq[iat])):
            pos = np.append(pos, np.array([[float(pip[iline][12:22]), float(pip[iline][25:35]), float(pip[iline][38:48])]]), axis=0)
            indif += 1
            iline += 1
        for i in range(indif - indifini):
            atomtype.append(pip[iline][0:2].replace(' ', ''))
            rmt.append(float(pip[iline][43:48]))
        iline += 4
    if ase:
        cell2 = coorsys(cell)
        atoms = Atoms(atomtype, pos, pbc=True)
        atoms.set_cell(cell2, scale_atoms=True)
        cell2 = np.dot(c2p(lattice), cell2)
        if lattice == 'R':
            atoms.set_cell(cell2, scale_atoms=True)
        else:
            atoms.set_cell(cell2)
        return atoms
    else:
        return (cell, lattice, pos, atomtype, rmt)