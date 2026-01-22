import numbers
import warnings
import numpy as np
from .kalman_smoother import KalmanSmoother
from .cfa_simulation_smoother import CFASimulationSmoother
from . import tools
@simulate_disturbance.setter
def simulate_disturbance(self, value):
    if bool(value):
        self.simulation_output = self.simulation_output | SIMULATION_DISTURBANCE
    else:
        self.simulation_output = self.simulation_output & ~SIMULATION_DISTURBANCE