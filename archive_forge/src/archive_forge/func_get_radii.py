from math import sqrt
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.data import covalent_radii
from ase.gui.defaults import read_defaults
from ase.io import read, write, string2index
from ase.gui.i18n import _
from ase.geometry import find_mic
import warnings
def get_radii(self, atoms):
    radii = np.array([self.covalent_radii[z] for z in atoms.numbers])
    radii *= self.atom_scale
    return radii