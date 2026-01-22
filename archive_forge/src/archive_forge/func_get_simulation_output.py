import numbers
import warnings
import numpy as np
from .kalman_smoother import KalmanSmoother
from .cfa_simulation_smoother import CFASimulationSmoother
from . import tools
def get_simulation_output(self, simulation_output=None, simulate_state=None, simulate_disturbance=None, simulate_all=None, **kwargs):
    """
        Get simulation output bitmask

        Helper method to get final simulation output bitmask from a set of
        optional arguments including the bitmask itself and possibly boolean
        flags.

        Parameters
        ----------
        simulation_output : int, optional
            Simulation output bitmask. If this is specified, it is simply
            returned and the other arguments are ignored.
        simulate_state : bool, optional
            Whether or not to include the state in the simulation output.
        simulate_disturbance : bool, optional
            Whether or not to include the state and observation disturbances
            in the simulation output.
        simulate_all : bool, optional
            Whether or not to include all simulation output.
        \\*\\*kwargs
            Additional keyword arguments. Present so that calls to this method
            can use \\*\\*kwargs without clearing out additional arguments.
        """
    if simulation_output is None:
        simulation_output = 0
        if simulate_state:
            simulation_output |= SIMULATION_STATE
        if simulate_disturbance:
            simulation_output |= SIMULATION_DISTURBANCE
        if simulate_all:
            simulation_output |= SIMULATION_ALL
        if simulation_output == 0:
            argument_set = not all([simulate_state is None, simulate_disturbance is None, simulate_all is None])
            if argument_set:
                raise ValueError('Invalid simulation output options: given options would result in no output.')
            simulation_output = self.smoother_output
    return simulation_output