import os
from os.path import dirname, abspath, join
import pyomo.common.unittest as unittest
import pyomo.opt
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
from pyomo.common.fileutils import import_file
from pyomo.core.base import Var
from pyomo.core.base.objective import minimize, maximize
from pyomo.core.base.piecewise import Bound, PWRepn
from pyomo.solvers.tests.solvers import test_solver_cases as _test_solver_cases
def assignProblems(cls, problem_list):
    for solver, writer in testing_solvers:
        for PROBLEM in problem_list:
            aux_list = ['', 'force_pw']
            for AUX in aux_list:
                for REPN in PWRepn:
                    for BOUND_TYPE in Bound:
                        for SENSE in [maximize, minimize]:
                            if not (BOUND_TYPE == Bound.Lower and SENSE == maximize or (BOUND_TYPE == Bound.Upper and SENSE == minimize) or (REPN in [PWRepn.BIGM_BIN, PWRepn.BIGM_SOS1, PWRepn.MC] and 'step' in PROBLEM)):
                                kwds = {}
                                kwds['sense'] = SENSE
                                kwds['pw_repn'] = REPN
                                kwds['pw_constr_type'] = BOUND_TYPE
                                if SENSE == maximize:
                                    attrName = 'test_{0}_{1}_{2}_{3}_{4}_{5}'.format(PROBLEM, REPN, BOUND_TYPE, 'maximize', solver, writer)
                                elif SENSE == minimize:
                                    attrName = 'test_{0}_{1}_{2}_{3}_{4}_{5}'.format(PROBLEM, REPN, BOUND_TYPE, 'minimize', solver, writer)
                                if AUX != '':
                                    kwds[AUX] = True
                                    attrName += '_' + AUX
                                setattr(cls, attrName, createMethod(attrName, PROBLEM, solver, writer, kwds))
                                if yaml_available:
                                    with open(join(thisDir, 'baselines', PROBLEM + '_baseline_results.yml'), 'r') as f:
                                        baseline_results = yaml.load(f, **yaml_load_args)
                                        setattr(cls, PROBLEM + '_results', baseline_results)