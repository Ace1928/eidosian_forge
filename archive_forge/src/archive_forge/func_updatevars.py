import numpy as np
from ase.md.md import MolecularDynamics
from ase.parallel import world, DummyMPI
from ase import units
def updatevars(self):
    dt = self.dt
    T = self.temp
    fr = self.fr
    masses = self.masses
    sigma = np.sqrt(2 * T * fr / masses)
    self.c1 = dt / 2.0 - dt * dt * fr / 8.0
    self.c2 = dt * fr / 2 - dt * dt * fr * fr / 8.0
    self.c3 = np.sqrt(dt) * sigma / 2.0 - dt ** 1.5 * fr * sigma / 8.0
    self.c5 = dt ** 1.5 * sigma / (2 * np.sqrt(3))
    self.c4 = fr / 2.0 * self.c5