import numpy as np
from ase.md.md import MolecularDynamics
from ase.parallel import world, DummyMPI
from ase import units
def set_friction(self, friction):
    self.fr = friction
    self.updatevars()