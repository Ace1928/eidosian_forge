from time import time
from qiskit.utils import optionals as _optionals
def recursiveBacktracking(self, solutions, domains, vconstraints, assignments, single):
    """Like ``constraint.RecursiveBacktrackingSolver.recursiveBacktracking`` but
            limited in the amount of calls by ``self.call_limit``"""
    if self.limit_reached():
        return None
    return super().recursiveBacktracking(solutions, domains, vconstraints, assignments, single)