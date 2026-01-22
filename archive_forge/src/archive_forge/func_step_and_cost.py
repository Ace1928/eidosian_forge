import numpy as np
import pennylane as qml
from pennylane.utils import _flatten, unflatten
def step_and_cost(self, objective_fn, x, generators, **kwargs):
    """Update trainable arguments with one step of the optimizer and return the corresponding
        objective function value prior to the step.

        Args:
            objective_fn (function): The objective function for optimization. It must have the
                signature ``objective_fn(x, generators=None)`` with a sequence of the values ``x``
                and a list of the gates ``generators`` as inputs, returning a single value.
            x (Union[Sequence[float], float]): sequence containing the initial values of the
                variables to be optimized over or a single float with the initial value
            generators (list[~.Operation]): list containing the initial ``pennylane.ops.qubit``
                operators to be used in the circuit and optimized over
            **kwargs : variable length of keyword arguments for the objective function.

        Returns:
            tuple: the new variable values :math:`x^{(t+1)}`, the new generators, and the objective
            function output prior to the step
        """
    x_new, generators = self.step(objective_fn, x, generators, **kwargs)
    return (x_new, generators, objective_fn(x, generators, **kwargs))