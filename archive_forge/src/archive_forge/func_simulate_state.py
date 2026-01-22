import numbers
import warnings
import numpy as np
from .kalman_smoother import KalmanSmoother
from .cfa_simulation_smoother import CFASimulationSmoother
from . import tools
@simulate_state.setter
def simulate_state(self, value):
    if bool(value):
        self.simulation_output = self.simulation_output | SIMULATION_STATE
    else:
        self.simulation_output = self.simulation_output & ~SIMULATION_STATE