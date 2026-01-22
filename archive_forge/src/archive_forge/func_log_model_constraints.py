import logging
from pyomo.core import Var, Constraint, TraversalStrategy
def log_model_constraints(m, logger=logger, active=True):
    """Prints the model constraints in the model."""
    for constr in m.component_data_objects(ctype=Constraint, active=active, descend_into=True, descent_order=TraversalStrategy.PrefixDepthFirstSearch):
        logger.info('%s %s' % (constr.name, 'active' if constr.active else 'deactivated'))