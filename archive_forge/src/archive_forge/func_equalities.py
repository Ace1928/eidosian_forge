from cvxopt.base import matrix, spmatrix
from cvxopt import blas, solvers 
import sys
def equalities(self):
    """ Returns a list of equality constraints of the LP."""
    return list(self._equalities)