import re
from collections import OrderedDict
import numpy as np
from ase import Atoms
from ase.units import Hartree, Bohr
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from .parser import _define_pattern
def parse_pw_chunk(chunk):
    atoms = _parse_geomblock(chunk)
    if atoms is None:
        return
    energy = None
    efermi = None
    forces = None
    stress = None
    matches = _nwpw_energy.findall(chunk)
    if matches:
        energy = float(matches[-1].replace('D', 'E')) * Hartree
    matches = _fermi_energy.findall(chunk)
    if matches:
        efermi = float(matches[-1].replace('D', 'E')) * Hartree
    gradblocks = _nwpw_grad.findall(chunk)
    if not gradblocks:
        gradblocks = _paw_grad.findall(chunk)
    if gradblocks:
        gradblock = gradblocks[-1].strip().split('\n')
        natoms = len(gradblock)
        symbols = []
        forces = np.zeros((natoms, 3))
        for i, line in enumerate(gradblock):
            line = line.strip().split()
            symbols.append(line[1])
            forces[i] = [float(x) for x in line[3:6]]
        forces *= Hartree / Bohr
    if atoms.cell:
        stress = _get_stress(chunk, atoms.cell)
    ibz_kpts, kpts = _get_pw_kpts(chunk)
    calc = SinglePointDFTCalculator(atoms=atoms, energy=energy, efermi=efermi, free_energy=energy, forces=forces, stress=stress, ibzkpts=ibz_kpts)
    calc.kpts = kpts
    atoms.calc = calc
    return atoms