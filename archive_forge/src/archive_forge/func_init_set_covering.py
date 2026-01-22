from contextlib import contextmanager
from math import fabs
from pyomo.common.collections import ComponentMap
from pyomo.contrib.gdpopt.cut_generation import add_no_good_cut
from pyomo.contrib.gdpopt.solve_discrete_problem import solve_MILP_discrete_problem
from pyomo.contrib.gdpopt.util import _DoNothing
from pyomo.core import Block, Constraint, Objective, Var, maximize, value
from pyomo.gdp import Disjunct
from pyomo.opt import TerminationCondition as tc
def init_set_covering(util_block, discrete_problem_util_block, subprob_util_block, config, solver):
    """Initialize by solving problems to cover the set of all disjuncts.

    The purpose of this initialization is to generate linearizations
    corresponding to each of the disjuncts.

    This work is based upon prototyping work done by Eloy Fernandez at
    Carnegie Mellon University.

    """
    config.logger.info('Starting set covering initialization.')
    solver._log_header(config.logger)
    disjunct_needs_cover = list((any((constr.body.polynomial_degree() not in (0, 1) for constr in disj.component_data_objects(ctype=Constraint, active=True, descend_into=True))) for disj in util_block.disjunct_list))
    subprob = subprob_util_block.parent_block()
    with use_discrete_problem_for_set_covering(discrete_problem_util_block):
        iter_count = 1
        while any(disjunct_needs_cover) and iter_count <= config.set_cover_iterlim:
            config.logger.debug('%s disjuncts need to be covered.' % disjunct_needs_cover.count(True))
            update_set_covering_objective(discrete_problem_util_block, disjunct_needs_cover)
            mip_termination = solve_MILP_discrete_problem(discrete_problem_util_block, solver, config)
            if mip_termination is tc.infeasible:
                config.logger.debug('Set covering problem is infeasible. Problem may have no more feasible disjunctive realizations.')
                if iter_count <= 1:
                    config.logger.warning('Set covering problem is infeasible. Check your linear and logical constraints for contradictions.')
                solver._update_dual_bound_to_infeasible()
                return False
            else:
                config.logger.debug('Solved set covering MIP')
            nlp_feasible = solver._fix_discrete_soln_solve_subproblem_and_add_cuts(discrete_problem_util_block, subprob_util_block, config)
            if nlp_feasible:
                active_disjuncts = list((fabs(value(disj.binary_indicator_var) - 1) <= config.integer_tolerance for disj in discrete_problem_util_block.disjunct_list))
                disjunct_needs_cover = list((needed_cover and (not was_active) for needed_cover, was_active in zip(disjunct_needs_cover, active_disjuncts)))
            add_no_good_cut(discrete_problem_util_block, config)
            iter_count += 1
            solver.initialization_iteration += 1
        if any(disjunct_needs_cover):
            config.logger.warning('Iteration limit reached for set covering initialization without covering all disjuncts.')
            return False
    config.logger.info('Initialization complete.')
    return True