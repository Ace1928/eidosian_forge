from pyomo.core import (
from pyomo.core.base import TransformationFactory, _VarData
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.common.config import ConfigBlock, ConfigValue, NonNegativeFloat
from pyomo.common.modeling import unique_component_name
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.opt import TerminationCondition
import logging
def post_process_fme_constraints(self, m, solver_factory, projected_constraints=None, tolerance=0):
    """Function that solves a sequence of LPs problems to check if
        constraints are implied by each other. Deletes any that are.

        Parameters
        ----------------
        m: A model, already transformed with FME. Note that if constraints
           have been added, activated, or deactivated, we will check for
           redundancy against the whole active part of the model. If you call
           this straight after FME, you are only checking within the projected
           constraints, but otherwise it is up to the user.
        solver_factory: A SolverFactory object (constructed with a solver
                        which can solve the continuous relaxation of the
                        active constraints on the model. That is, if you
                        had nonlinear constraints unrelated to the variables
                        being projected, you need to either deactivate them or
                        provide a solver which will do the right thing.)
        projected_constraints: The ConstraintList of projected constraints.
                               Default is None, in which case we assume that
                               the FME transformation was called without
                               specifying their name, so will look for them on
                               the private transformation block.
        tolerance: Tolerance at which we decide a constraint is implied by the
                   others. Default is 0, meaning we remove the constraint if
                   the LP solve finds the constraint can be tight but not
                   violated. Setting this to a small positive value would
                   remove constraints more conservatively. Setting it to a
                   negative value would result in a relaxed problem.
        """
    if projected_constraints is None:
        if not hasattr(m, '_pyomo_contrib_fme_transformation'):
            raise RuntimeError('It looks like model %s has not been transformed with the fourier_motzkin_elimination transformation!' % m.name)
        transBlock = m._pyomo_contrib_fme_transformation
        if not hasattr(transBlock, 'projected_constraints'):
            raise RuntimeError("It looks the projected constraints were manually named when the FME transformation was called on %s. If this is so, specify the ConstraintList of projected constraints with the 'projected_constraints' argument." % m.name)
        projected_constraints = transBlock.projected_constraints
    TransformationFactory('core.relax_integer_vars').apply_to(m, transform_deactivated_blocks=True)
    active_objs = []
    for obj in m.component_data_objects(Objective, descend_into=True):
        if obj.active:
            active_objs.append(obj)
        obj.deactivate()
    obj_name = unique_component_name(m, '_fme_post_process_obj')
    obj = Objective(expr=0)
    m.add_component(obj_name, obj)
    for i in projected_constraints:
        if not projected_constraints[i].active:
            continue
        projected_constraints[i].deactivate()
        m.del_component(obj)
        obj = Objective(expr=projected_constraints[i].body - projected_constraints[i].lower)
        m.add_component(obj_name, obj)
        results = solver_factory.solve(m)
        if results.solver.termination_condition == TerminationCondition.unbounded:
            obj_val = -float('inf')
        elif results.solver.termination_condition != TerminationCondition.optimal:
            raise RuntimeError('Unsuccessful subproblem solve when checkingconstraint %s.\n\tTermination Condition: %s' % (projected_constraints[i].name, results.solver.termination_condition))
        else:
            obj_val = value(obj)
        if obj_val >= tolerance:
            m.del_component(projected_constraints[i])
            del projected_constraints[i]
        else:
            projected_constraints[i].activate()
    m.del_component(obj)
    for obj in active_objs:
        obj.activate()
    TransformationFactory('core.relax_integer_vars').apply_to(m, undo=True)