import ctypes
import numpy as np
from numpy.linalg import norm
from ase.calculators.calculator import Calculator
from ase.data import (atomic_numbers as ase_atomic_numbers,
from ase.calculators.lammps import convert
from ase.geometry import wrap_positions
def restart_lammps(self, atoms):
    if self.started:
        self.lmp.command('clear')
    self.started = False
    self.initialized = False
    self.previous_atoms_numbers = []
    self.start_lammps()
    self.initialise_lammps(atoms)