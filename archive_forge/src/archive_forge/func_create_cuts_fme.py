from pyomo.common.config import (
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.expr import differentiate
from pyomo.common.collections import ComponentSet
from pyomo.opt import SolverFactory
from pyomo.repn import generate_standard_repn
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import (
from pyomo.contrib.fme.fourier_motzkin_elimination import (
import logging
def create_cuts_fme(transBlock_rHull, var_info, hull_to_bigm_map, rBigM_linear_constraints, rHull_vars, disaggregated_vars, norm, cut_threshold, zero_tolerance, integer_arithmetic, constraint_tolerance):
    """Returns a cut which removes x* from the relaxed bigm feasible region.

    Finds all the constraints which are tight at xhat (assumed to be the
    solution currently in instance_rHull), and calculates a composite normal
    vector by summing the vectors normal to each of these constraints. Then
    Fourier-Motzkin elimination is used to project the disaggregated variables
    out of the polyhedron formed by the composite normal and the collection
    of tight constraints. This results in multiple cuts, of which we select
    one that cuts of x* by the greatest margin, as long as that margin is
    more than cut_threshold. If no cut satisfies the margin specified by
    cut_threshold, we return None.

    Parameters
    -----------
    transBlock_rHull: transformation block on relaxed hull instance
    var_info: List of tuples (rBigM_var, rHull_var, xstar_param)
    hull_to_bigm_map: For expression substitution, maps id(hull_var) to
                      corresponding bigm var
    rBigM_linear_constraints: list of linear constraints in relaxed bigM
    rHull_vars: list of all variables in relaxed hull
    disaggregated_vars: ComponentSet of disaggregated variables in hull
                        reformulation
    norm: norm used in the separation problem
    cut_threshold: Amount x* needs to be infeasible in generated cut in order
                   to consider the cut for addition to the bigM model.
    zero_tolerance: Tolerance at which a float will be treated as 0 during
                    Fourier-Motzkin elimination
    integer_arithmetic: boolean, whether or not to require Fourier-Motzkin
                        Elimination does integer arithmetic. Only possible
                        when all data is integer.
    constraint_tolerance: Tolerance at which we will consider a constraint
                          tight.
    """
    instance_rHull = transBlock_rHull.model()
    if transBlock_rHull.component('constraints_for_FME') is None:
        _precompute_potentially_useful_constraints(transBlock_rHull, disaggregated_vars)
    tight_constraints = Block()
    conslist = tight_constraints.constraints = Constraint(NonNegativeIntegers)
    conslist.construct()
    something_interesting = False
    for constraint in transBlock_rHull.constraints_for_FME:
        multipliers = _constraint_tight(instance_rHull, constraint, constraint_tolerance)
        for multiplier in multipliers:
            if multiplier:
                something_interesting = True
                f = constraint.body
                firstDerivs = differentiate(f, wrt_list=rHull_vars)
                normal_vec = [multiplier * value(_) for _ in firstDerivs]
                if f.polynomial_degree() == 1:
                    conslist[len(conslist)] = constraint.expr
                else:
                    conslist[len(conslist)] = _get_linear_approximation_expr(normal_vec, rHull_vars)
    if not something_interesting:
        return None
    tight_constraints.construct()
    logger.info('Calling FME transformation on %s constraints to eliminate %s variables' % (len(tight_constraints.constraints), len(disaggregated_vars)))
    TransformationFactory('contrib.fourier_motzkin_elimination').apply_to(tight_constraints, vars_to_eliminate=disaggregated_vars, zero_tolerance=zero_tolerance, do_integer_arithmetic=integer_arithmetic, projected_constraints_name='fme_constraints')
    fme_results = tight_constraints.fme_constraints
    projected_constraints = [cons for i, cons in fme_results.items()]
    cuts = _get_constraint_exprs(projected_constraints, hull_to_bigm_map)
    best = 0
    best_cut = None
    cuts_to_keep = []
    for i, cut in enumerate(cuts):
        logger.info('FME: Post-processing cut %s' % cut)
        if value(cut):
            logger.info("FME:\t Doesn't cut off x*")
            continue
        cuts_to_keep.append(i)
        assert len(cut.args) == 2
        cut_off = value(cut.args[0]) - value(cut.args[1])
        if cut_off > cut_threshold and cut_off > best:
            best = cut_off
            best_cut = cut
            logger.info('FME:\t New best cut: Cuts off x* by %s.' % best)
    cuts = [cuts[i] for i in cuts_to_keep]
    if best_cut is not None:
        return [best_cut]
    return None