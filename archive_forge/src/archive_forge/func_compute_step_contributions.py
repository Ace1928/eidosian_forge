import numpy as np
from ase.optimize.optimize import Dynamics
def compute_step_contributions(self, potentiostat_step_size):
    """Computes the orthogonal component sizes of the step so that the net
        step obeys the smaller of step_size or maxstep."""
    if abs(potentiostat_step_size) < self.step_size:
        delta_s_perpendicular = potentiostat_step_size
        contour_step_size = np.sqrt(self.step_size ** 2 - potentiostat_step_size ** 2)
        delta_s_parallel = np.sqrt(1 - self.parallel_drift ** 2) * contour_step_size
        delta_s_drift = contour_step_size * self.parallel_drift
    else:
        delta_s_parallel = 0.0
        delta_s_drift = 0.0
        delta_s_perpendicular = np.sign(potentiostat_step_size) * self.step_size
    return (delta_s_perpendicular, delta_s_parallel, delta_s_drift)