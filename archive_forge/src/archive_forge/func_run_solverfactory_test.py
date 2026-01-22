import sys
from io import StringIO
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.repn.tests.lp_diff import lp_diff
def run_solverfactory_test():
    skip_solvers = {'py', 'xpress', '_xpress_shell', '_mock_xpress'}
    with LoggingIntercept() as LOG, capture_output(capture_fd=True) as OUT:
        info = []
        for solver in sorted(pyo.SolverFactory):
            _doc = pyo.SolverFactory.doc(solver)
            if _doc is not None and 'DEPRECATED' in _doc:
                _avail = 'DEPR'
            elif solver in skip_solvers:
                _avail = 'SKIP'
            else:
                _avail = str(pyo.SolverFactory(solver).available(False))
            info.append('   %s(%s): %s' % (solver, _avail, _doc))
        glpk = pyo.SolverFactory('glpk')
    print('')
    print('Pyomo Solvers')
    print('-------------')
    print('\n'.join(info))
    if type(glpk.available(False)) != bool:
        print('Solver glpk.available() did not return bool')
        sys.exit(3)
    _check_log_and_out(LOG, OUT, 20)