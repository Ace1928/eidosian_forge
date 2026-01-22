import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
def variable_values(self, variables=None):
    """The variable values from the best primal feasible solution.

        An error will be raised if there are no primal feasible solutions.

        Args:
          variables: an optional Variable or iterator of Variables indicating what
            variable values to return. If not provided, variable_values returns a
            dictionary with all the variable values for all variables.

        Returns:
          The variable values from the best primal feasible solution.

        Raises:
          ValueError: There are no primal feasible solutions.
          TypeError: Argument is not None, a Variable or an iterable of Variables.
          KeyError: Variable values requested for an invalid variable (e.g. is not a
            Variable or is a variable for another model).
        """
    if not self.has_primal_feasible_solution():
        raise ValueError('No primal feasible solution available.')
    assert self.solutions[0].primal_solution is not None
    if variables is None:
        return self.solutions[0].primal_solution.variable_values
    if isinstance(variables, model.Variable):
        return self.solutions[0].primal_solution.variable_values[variables]
    if isinstance(variables, Iterable):
        return [self.solutions[0].primal_solution.variable_values[v] for v in variables]
    raise TypeError(f'unsupported type in argument for variable_values: {type(variables).__name__!r}')