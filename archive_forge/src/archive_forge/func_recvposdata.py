import os
import socket
from subprocess import Popen, PIPE
from contextlib import contextmanager
import numpy as np
from ase.calculators.calculator import (Calculator, all_changes,
import ase.units as units
from ase.utils import IOContext
from ase.stress import full_3x3_to_voigt_6_stress
def recvposdata(self):
    cell = self.recv((3, 3), np.float64).T.copy()
    icell = self.recv((3, 3), np.float64).T.copy()
    natoms = self.recv(1, np.int32)
    natoms = int(natoms)
    positions = self.recv((natoms, 3), np.float64)
    return (cell * units.Bohr, icell / units.Bohr, positions * units.Bohr)