import numpy as np
from ase.optimize.optimize import Dynamics
def compute_potentiostat_step_size(self, forces, energy):
    """Computes the potentiostat step size by linear extrapolation of the
        potential energy using the forces. The step size can be positive or
        negative depending on whether or not the energy is too high or too low.
        """
    if self.use_target_shift:
        target_shift = self.energy_target - np.mean(self.previous_energies)
    else:
        target_shift = 0.0
    deltaU = energy - (self.energy_target + target_shift)
    f_norm = np.linalg.norm(forces)
    potentiostat_step_size = deltaU / f_norm * self.potentiostat_step_scale
    return potentiostat_step_size