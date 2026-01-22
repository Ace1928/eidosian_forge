import sys
import logging
from pyomo.common.collections import Bunch
from pyomo.opt import TerminationCondition
from pyomo.solvers.tests.models.base import all_models
from pyomo.solvers.tests.solvers import test_solver_cases as _test_solver_cases
from pyomo.core.kernel.block import IBlock
def run_scenarios(options):
    logging.disable(logging.WARNING)
    solvers = set(options.solver)
    stat = {}
    for key, test_case in generate_scenarios():
        model, solver, io = key
        if len(solvers) > 0 and (not solver in solvers):
            continue
        if test_case.status == 'skip':
            continue
        model_class = test_case.model()
        model_class.generate_model()
        model_class.warmstart_model()
        symbolic_labels = False
        load_solutions = False
        opt, results = model_class.solve(solver, io, test_case.testcase.io_options, {}, symbolic_labels, load_solutions)
        termination_condition = results['Solver'][0]['termination condition']
        try:
            model_class.post_solve_test_validation(None, results)
        except:
            if test_case.status == 'expected failure':
                stat[key] = (True, 'Expected failure')
            else:
                stat[key] = (False, 'Unexpected termination condition: %s' % str(termination_condition))
            continue
        if termination_condition == TerminationCondition.unbounded or termination_condition == TerminationCondition.infeasible:
            stat[key] = (True, '')
        else:
            if isinstance(model_class.model, IBlock):
                model_class.model.load_solution(results.solution)
            else:
                model_class.model.solutions.load_from(results, default_variable_value=opt.default_variable_value())
            rc = model_class.validate_current_solution(suffixes=model_class.test_suffixes)
            if test_case.status == 'expected failure':
                if rc[0] is True:
                    stat[key] = (False, 'Unexpected success')
                else:
                    stat[key] = (True, 'Expected failure')
            elif rc[0] is True:
                stat[key] = (True, '')
            else:
                stat[key] = (False, 'Unexpected failure')
    if options.verbose:
        print('---------------')
        print(' Test Failures')
        print('---------------')
    nfail = 0
    summary = {}
    total = Bunch(NumEPass=0, NumEFail=0, NumUPass=0, NumUFail=0)
    for key in stat:
        model, solver, io = key
        if not solver in summary:
            summary[solver] = Bunch(NumEPass=0, NumEFail=0, NumUPass=0, NumUFail=0)
        _pass, _str = stat[key]
        if _pass:
            if _str == 'Expected failure':
                summary[solver].NumEFail += 1
            else:
                summary[solver].NumEPass += 1
        else:
            nfail += 1
            if _str == 'Unexpected failure':
                summary[solver].NumUFail += 1
                if options.verbose:
                    print('- Unexpected Test Failure: ' + ', '.join((model, solver, io)))
            else:
                summary[solver].NumUPass += 1
                if options.verbose:
                    print('- Unexpected Test Success: ' + ', '.join((model, solver, io)))
    if options.verbose:
        if nfail == 0:
            print('- NONE')
        print('')
    stream = sys.stdout
    maxSolverNameLen = max([max((len(name) for name in summary)), len('Solver')])
    fmtStr = '{{0:<{0}}}| {{1:>8}} | {{2:>8}} | {{3:>10}} | {{4:>10}} | {{5:>13}}\n'.format(maxSolverNameLen + 2)
    stream.write('\n')
    stream.write('Solver Test Summary\n')
    stream.write('=' * (maxSolverNameLen + 66) + '\n')
    stream.write(fmtStr.format('Solver', '# Pass', '# Fail', '# OK Fail', '# Bad Pass', '% OK'))
    stream.write('=' * (maxSolverNameLen + 66) + '\n')
    for _solver in sorted(summary):
        ans = summary[_solver]
        total.NumEPass += ans.NumEPass
        total.NumEFail += ans.NumEFail
        total.NumUPass += ans.NumUPass
        total.NumUFail += ans.NumUFail
        stream.write(fmtStr.format(_solver, str(ans.NumEPass), str(ans.NumUFail), str(ans.NumEFail), str(ans.NumUPass), str(int(100.0 * (ans.NumEPass + ans.NumEFail) / (ans.NumEPass + ans.NumEFail + ans.NumUFail + ans.NumUPass)))))
    stream.write('=' * (maxSolverNameLen + 66) + '\n')
    stream.write(fmtStr.format('TOTALS', str(total.NumEPass), str(total.NumUFail), str(total.NumEFail), str(total.NumUPass), str(int(100.0 * (total.NumEPass + total.NumEFail) / (total.NumEPass + total.NumEFail + total.NumUFail + total.NumUPass)))))
    stream.write('=' * (maxSolverNameLen + 66) + '\n')
    logging.disable(logging.NOTSET)