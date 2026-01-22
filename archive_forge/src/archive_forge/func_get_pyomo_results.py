from collections import namedtuple
from pyomo.core.base.objective import Objective
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.modeling import unique_component_name
from pyomo.common.config import ConfigBlock, ConfigValue, In
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.opt import SolverResults, TerminationCondition
from pyomo.common.dependencies import (
def get_pyomo_results(self, model, scipy_results):
    nlp = self.get_nlp()
    results = SolverResults()
    if self._nlp_solver.options.full_output:
        root, res = scipy_results
    else:
        root = scipy_results
    results.problem.name = model.name
    results.problem.number_of_constraints = nlp.n_eq_constraints()
    results.problem.number_of_variables = nlp.n_primals()
    results.problem.number_of_binary_variables = 0
    results.problem.number_of_integer_variables = 0
    results.problem.number_of_continuous_variables = nlp.n_primals()
    results.solver.name = self._solver_name
    results.solver.wallclock_time = self._timer.timers['solve'].total_time
    if self._nlp_solver.options.full_output:
        results.solver.message = res.flag
        if res.converged:
            term_cond = TerminationCondition.feasible
        else:
            term_cond = TerminationCondition.Error
        results.solver.termination_condition = term_cond
        results.solver.status = TerminationCondition.to_solver_status(results.solver.termination_condition)
        results.solver.number_of_function_evaluations = res.function_calls
    return results