from typing import List, Union
import cvxpy.atoms as atoms
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.transforms import indicator
def log_sum_exp(objectives: List[Union[Minimize, Maximize]], weights, gamma: float=1.0) -> Minimize:
    """Combines objectives as log_sum_exp of weighted terms.


    The objective takes the form
        log(sum_{i=1}^n exp(gamma*weights[i]*objectives[i]))/gamma
    As gamma goes to 0, log_sum_exp approaches weighted_sum. As gamma goes to infinity,
    log_sum_exp approaches max.

    Args:
      objectives: A list of Minimize/Maximize objectives.
      weights: A vector of weights.
      gamma: Parameter interpolating between weighted_sum and max.

    Returns:
      A Minimize objective.
    """
    num_objs = len(objectives)
    terms = [(objectives[i] * weights[i]).args[0] for i in range(num_objs)]
    expr = atoms.log_sum_exp(gamma * atoms.vstack(terms)) / gamma
    return Minimize(expr)