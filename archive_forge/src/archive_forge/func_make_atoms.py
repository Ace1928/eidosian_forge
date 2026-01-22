import numpy as np
import pytest
from ase.data import s22
from ase.optimize import FIRE
from ase.constraints import FixBondLengths
from ase.calculators.tip3p import TIP3P, epsilon0, sigma0
from ase.calculators.combine_mm import CombineMM
def make_atoms():
    atoms = s22.create_s22_system('Water_dimer')
    center = atoms[0].position
    atoms.translate(-center)
    h = atoms[3].position[1] - atoms[0].position[1]
    l = np.linalg.norm(atoms[0].position - atoms[3].position)
    angle = np.degrees(np.arcsin(h / l))
    atoms.rotate(angle, '-z', center=center)
    return atoms