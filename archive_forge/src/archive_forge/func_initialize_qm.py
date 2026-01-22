import numpy as np
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers
from ase.utils import IOContext
from ase.geometry import get_distances
from ase.cell import Cell
def initialize_qm(self, atoms):
    constraints = atoms.constraints
    atoms.constraints = []
    self.qmatoms = atoms[self.selection]
    atoms.constraints = constraints
    self.qmatoms.pbc = False
    if self.vacuum:
        self.qmatoms.center(vacuum=self.vacuum)
        self.center = self.qmatoms.positions.mean(axis=0)