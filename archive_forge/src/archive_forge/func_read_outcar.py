import os
import sys
import re
import numpy as np
import subprocess
from contextlib import contextmanager
from pathlib import Path
from warnings import warn
from typing import Dict, Any
from xml.etree import ElementTree
import ase
from ase.io import read, jsonio
from ase.utils import PurePath
from ase.calculators import calculator
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.calculators.vasp.create_input import GenerateVaspInput
def read_outcar(self, lines=None):
    """Read results from the OUTCAR file.
        Deprecated, see read_results()"""
    if not lines:
        lines = self.load_file('OUTCAR')
    self.spinpol = self.get_spin_polarized()
    self.version = self.get_version()
    self.energy_free, self.energy_zero = self.read_energy(lines=lines)
    self.forces = self.read_forces(lines=lines)
    self.fermi = self.read_fermi(lines=lines)
    self.dipole = self.read_dipole(lines=lines)
    self.stress = self.read_stress(lines=lines)
    self.nbands = self.read_nbands(lines=lines)
    self.read_ldau()
    self.magnetic_moment, self.magnetic_moments = self.read_mag(lines=lines)