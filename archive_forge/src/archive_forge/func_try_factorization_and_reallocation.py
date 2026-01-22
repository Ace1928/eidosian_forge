from pyomo.contrib.pynumero.interfaces.utils import (
import numpy as np
import logging
import time
from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus
from pyomo.common.timing import HierarchicalTimer
import enum
def try_factorization_and_reallocation(kkt, linear_solver, reallocation_factor, max_iter, timer=None):
    if timer is None:
        timer = HierarchicalTimer()
    assert max_iter >= 1
    for count in range(max_iter):
        timer.start('symbolic')
        '\n        Performance could be improved significantly by only performing \n        symbolic factorization once.\n\n        However, we first have to make sure the nonzero structure \n        (and ordering of row and column arrays) of the KKT matrix never \n        changes. We have not had time to test this thoroughly, yet. \n        '
        res = linear_solver.do_symbolic_factorization(matrix=kkt, raise_on_error=False)
        timer.stop('symbolic')
        if res.status == LinearSolverStatus.successful:
            timer.start('numeric')
            res = linear_solver.do_numeric_factorization(matrix=kkt, raise_on_error=False)
            timer.stop('numeric')
        status = res.status
        if status == LinearSolverStatus.not_enough_memory:
            linear_solver.increase_memory_allocation(reallocation_factor)
        else:
            break
    return (status, count)