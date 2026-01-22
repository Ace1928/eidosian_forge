import json
import os
from os.path import dirname, abspath, join
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import import_file
from pyomo.kernel import SolverFactory, variable, maximize, minimize
from pyomo.solvers.tests.solvers import test_solver_cases as _test_solver_cases
def createTestMethod(pName, problem, solver, writer, kwds):

    def testMethod(obj):
        if not testing_solvers[solver, writer]:
            obj.skipTest('Solver %s (interface=%s) is not available' % (solver, writer))
        m = import_file(os.path.join(thisDir, 'kernel_problems', problem + '.py'), clear_cache=True)
        model = m.define_model(**kwds)
        opt = SolverFactory(solver, solver_io=writer)
        results = opt.solve(model)
        new_results = ((var.name, var.value) for var in model.components(ctype=variable.ctype, active=True, descend_into=False))
        baseline_results = getattr(obj, problem + '_results')
        for name, value in new_results:
            if abs(baseline_results[name] - value) > 1e-05:
                raise IOError('Difference in baseline solution values and current solution values using:\n' + 'Solver: ' + solver + '\n' + 'Writer: ' + writer + '\n' + 'Variable: ' + name + '\n' + 'Solution: ' + str(value) + '\n' + 'Baseline: ' + str(baseline_results[name]) + '\n')
    return testMethod