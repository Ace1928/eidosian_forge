import logging
import random
from pyomo.core import Var
def reinitialize_variables(model, config):
    """Reinitializes all variable values in the model.

    Excludes fixed, noncontinuous, and unbounded variables.

    """
    for var in model.component_data_objects(ctype=Var, descend_into=True):
        if var.is_fixed() or not var.is_continuous():
            continue
        if var.lb is None or var.ub is None:
            if not config.suppress_unbounded_warning:
                logger.warning('Skipping reinitialization of unbounded variable %s with bounds (%s, %s). To suppress this message, set the suppress_unbounded_warning flag.' % (var.name, var.lb, var.ub))
            continue
        val = var.value if var.value is not None else (var.lb + var.ub) / 2
        var.set_value(strategies[config.strategy](val, var.lb, var.ub), skip_validation=True)