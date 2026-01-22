from .polynomial import Polynomial
from .component import NonZeroDimensionalComponent
from ..pari import pari
def numerical_solutions_with_one(polys):
    solutions = numerical_solutions(polys)
    for solution in solutions:
        if not isinstance(solution, NonZeroDimensionalComponent):
            solution['1'] = pari(1)
    return solutions