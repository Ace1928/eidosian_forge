import os
import re
import warnings
import numpy as np
from copy import deepcopy
import ase
from ase.parallel import paropen
from ase.spacegroup import Spacegroup
from ase.geometry.cell import cellpar_to_cell
from ase.constraints import FixAtoms, FixedPlane, FixedLine, FixCartesian
from ase.utils import atoms_to_spglib_cell
import ase.units
def read_castep_phonon(fd, index=None, read_vib_data=False, gamma_only=True, frequency_factor=None, units=units_CODATA2002):
    """
    Reads a .phonon file written by a CASTEP Phonon task and returns an atoms
    object, as well as the calculated vibrational data if requested.

    Note that the index argument has no effect as of now.
    """
    lines = fd.readlines()
    atoms = None
    cell = []
    N = Nb = Nq = 0
    scaled_positions = []
    symbols = []
    masses = []
    L = 0
    while L < len(lines):
        line = lines[L]
        if 'Number of ions' in line:
            N = int(line.split()[3])
        elif 'Number of branches' in line:
            Nb = int(line.split()[3])
        elif 'Number of wavevectors' in line:
            Nq = int(line.split()[3])
        elif 'Unit cell vectors (A)' in line:
            for ll in range(3):
                L += 1
                fields = lines[L].split()
                cell.append([float(x) for x in fields[0:3]])
        elif 'Fractional Co-ordinates' in line:
            for ll in range(N):
                L += 1
                fields = lines[L].split()
                scaled_positions.append([float(x) for x in fields[1:4]])
                symbols.append(fields[4])
                masses.append(float(fields[5]))
        elif 'END header' in line:
            L += 1
            atoms = ase.Atoms(symbols=symbols, scaled_positions=scaled_positions, cell=cell)
            break
        L += 1
    if frequency_factor is None:
        Kayser_to_eV = 100.0 * 2 * np.pi * units['hbar'] * units['c']
    frequency_factor = Kayser_to_eV
    qpoints = []
    weights = []
    frequencies = []
    displacements = []
    for nq in range(Nq):
        fields = lines[L].split()
        qpoints.append([float(x) for x in fields[2:5]])
        weights.append(float(fields[5]))
    freqs = []
    for ll in range(Nb):
        L += 1
        fields = lines[L].split()
        freqs.append(frequency_factor * float(fields[1]))
    frequencies.append(np.array(freqs))
    L += 2
    disps = []
    for ll in range(Nb):
        disp_coords = []
        for lll in range(N):
            L += 1
            fields = lines[L].split()
            disp_x = float(fields[2]) + float(fields[3]) * 1j
            disp_y = float(fields[4]) + float(fields[5]) * 1j
            disp_z = float(fields[6]) + float(fields[7]) * 1j
            disp_coords.extend([disp_x, disp_y, disp_z])
        disps.append(np.array(disp_coords))
    displacements.append(np.array(disps))
    if read_vib_data:
        if gamma_only:
            vibdata = [frequencies[0], displacements[0]]
        else:
            vibdata = [qpoints, weights, frequencies, displacements]
        return (vibdata, atoms)
    else:
        return atoms