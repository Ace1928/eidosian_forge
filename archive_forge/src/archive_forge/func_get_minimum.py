import numpy as np
from ase.optimize.optimize import Dynamics
from ase.optimize.fire import FIRE
from ase.units import kB
from ase.parallel import world
from ase.io.trajectory import Trajectory
def get_minimum(self):
    """Return minimal energy and configuration."""
    atoms = self.atoms.copy()
    atoms.set_positions(self.rmin)
    return (self.Emin, atoms)