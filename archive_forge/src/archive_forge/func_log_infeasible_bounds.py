from pyomo.core import Constraint, Var, value
from math import fabs
import logging
from pyomo.common import deprecated
from pyomo.core.expr.visitor import identify_variables
from pyomo.util.blockutil import log_model_constraints
def log_infeasible_bounds(m, tol=1e-06, logger=logger):
    """Logs the infeasible variable bounds in the model.

    Parameters
    ----------
    m: Block
        Pyomo block or model to check

    tol: float
        absolute feasibility tolerance

    logger: logging.Logger
        Logger to output to; defaults to `pyomo.util.infeasible`.

    """
    if logger.getEffectiveLevel() > logging.INFO:
        logger.warning('log_infeasible_bounds() called with a logger whose effective level is higher than logging.INFO: no output will be logged regardless of bound feasibility')
    for var, infeas in find_infeasible_bounds(m, tol):
        if infeas & 4:
            logger.info(f'VAR {var.name}: {_evaluation_errors[infeas]}.')
            continue
        if infeas & 1:
            logger.info(f'VAR {var.name}: LB {var.lb} </= {var.value}')
        if infeas & 2:
            logger.info(f'VAR {var.name}: {var.value} </= UB {var.ub}')