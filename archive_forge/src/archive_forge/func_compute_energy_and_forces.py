import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes
def compute_energy_and_forces(self, atoms):
    disps = atoms.positions - self.ideal_positions
    forces = -self.k * disps
    energy = sum(self.k / 2.0 * np.linalg.norm(disps, axis=1) ** 2)
    return (energy, forces)