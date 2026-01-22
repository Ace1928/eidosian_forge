from . import solutionsToPrimeIdealGroebnerBasis
from . import numericalSolutionsToGroebnerBasis
from .component import *
from .coordinates import PtolemyCoordinates
def process_solution(solution):
    assert isinstance(solution, dict)
    return PtolemyCoordinates(solution, is_numerical=True, py_eval_section=self.py_eval, manifold_thunk=self.manifold_thunk)