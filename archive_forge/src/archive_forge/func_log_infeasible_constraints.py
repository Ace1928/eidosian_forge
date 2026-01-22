from pyomo.core import Constraint, Var, value
from math import fabs
import logging
from pyomo.common import deprecated
from pyomo.core.expr.visitor import identify_variables
from pyomo.util.blockutil import log_model_constraints
def log_infeasible_constraints(m, tol=1e-06, logger=logger, log_expression=False, log_variables=False):
    """Logs the infeasible constraints in the model.

    Uses the current model state.  Messages are logged at the INFO level.

    Parameters
    ----------
    m: Block
        Pyomo block or model to check

    tol: float
        absolute feasibility tolerance

    logger: logging.Logger
        Logger to output to; defaults to `pyomo.util.infeasible`.

    log_expression: bool
        If true, prints the constraint expression

    log_variables: bool
        If true, prints the constraint variable names and values

    """
    if logger.getEffectiveLevel() > logging.INFO:
        logger.warning('log_infeasible_constraints() called with a logger whose effective level is higher than logging.INFO: no output will be logged regardless of constraint feasibility')
    for constr, body, infeas in find_infeasible_constraints(m, tol):
        if constr.equality:
            lb = lb_expr = lb_op = ''
            ub_expr = constr.upper
            ub = value(ub_expr, exception=False)
            if body is None:
                ub_op = ' =?= '
            else:
                ub_op = ' =/= '
        else:
            if constr.has_lb():
                lb_expr = constr.lower
                lb = value(lb_expr, exception=False)
                if body is None:
                    lb_op = ' <?= '
                elif infeas & 1:
                    lb_op = ' </= '
                else:
                    lb_op = ' <= '
            else:
                lb = lb_expr = lb_op = ''
            if constr.has_ub():
                ub_expr = constr.upper
                ub = value(ub_expr, exception=False)
                if body is None:
                    ub_op = ' <?= '
                elif infeas & 2:
                    ub_op = ' </= '
                else:
                    ub_op = ' <= '
            else:
                ub = ub_expr = ub_op = ''
        if body is None:
            body = 'evaluation error'
        if lb is None:
            lb = 'evaluation error'
        if ub is None:
            ub = 'evaluation error'
        line = f'CONSTR {constr.name}: {lb}{lb_op}{body}{ub_op}{ub}'
        if log_expression:
            line += f'\n  - EXPR: {lb_expr}{lb_op}{constr.body}{ub_op}{ub_expr}'
        if log_variables:
            line += ''.join((f'\n  - VAR {v.name}: {v.value}' for v in identify_variables(constr.body, include_fixed=True)))
        logger.info(line)