import os
from os.path import join, dirname, abspath
import types
import pyomo.common.unittest as unittest
from pyomo.opt import TerminationCondition
from pyomo.solvers.tests.models.base import all_models
from pyomo.solvers.tests.testcases import generate_scenarios
from pyomo.core.kernel.block import IBlock
def writer_test(self):
    model_class = test_case.model()
    save_filename = join(thisDir, '%s.soln.json' % model_class.description)
    if os.path.exists(save_filename):
        os.remove(save_filename)
    model_class.generate_model(test_case.testcase.import_suffixes)
    model_class.warmstart_model()
    load_solutions = False
    opt, results = model_class.solve(solver, io, test_case.testcase.io_options, test_case.testcase.options, symbolic_labels, load_solutions)
    termination_condition = results['Solver'][0]['termination condition']
    model_class.post_solve_test_validation(self, results)
    if termination_condition == TerminationCondition.unbounded or termination_condition == TerminationCondition.infeasible or termination_condition == TerminationCondition.infeasibleOrUnbounded:
        return
    if isinstance(model_class.model, IBlock):
        model_class.model.load_solution(results.Solution)
    else:
        model_class.model.solutions.load_from(results, default_variable_value=opt.default_variable_value())
        model_class.save_current_solution(save_filename, suffixes=model_class.test_suffixes)
    rc = model_class.validate_current_solution(suffixes=model_class.test_suffixes, exclude_suffixes=test_case.exclude_suffixes)
    if is_expected_failure:
        if rc[0]:
            self.fail("\nTest model '%s' was marked as an expected failure but no failure occurred. The reason given for the expected failure is:\n\n****\n%s\n****\n\nPlease remove this case as an expected failure if the above issue has been corrected in the latest version of the solver." % (model_class.description, test_case.msg))
        if _cleanup_expected_failures:
            os.remove(save_filename)
    if not rc[0]:
        if not isinstance(model_class.model, IBlock):
            try:
                model_class.model.solutions.store_to(results)
            except ValueError:
                pass
        self.fail('Solution mismatch for plugin ' + test_name + ', ' + io + ' interface and problem type ' + model_class.description + '\n' + rc[1] + '\n' + (str(results.Solution(0)) if len(results.solution) else 'No Solution'))
    try:
        os.remove(save_filename)
    except OSError:
        pass