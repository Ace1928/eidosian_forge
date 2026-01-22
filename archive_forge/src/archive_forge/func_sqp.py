from pyomo.contrib.pynumero.interfaces.nlp import NLP
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
from pyomo.contrib.pynumero.linalg.ma27_interface import MA27
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import numpy as np
from scipy.sparse import tril
import pyomo.environ as pe
from pyomo import dae
from pyomo.common.timing import TicTocTimer
import time
from pyomo.contrib.pynumero.linalg.base import LinearSolverInterface, LinearSolverStatus
def sqp(nlp: NLP, linear_solver: LinearSolverInterface, max_iter=100, tol=1e-08, output=True):
    """
    An example of a simple SQP algorithm for
    equality-constrained NLPs.

    Parameters
    ----------
    nlp: NLP
        A PyNumero NLP
    max_iter: int
        The maximum number of iterations
    tol: float
        The convergence tolerance
    """
    t0 = time.time()
    kkt = BlockMatrix(2, 2)
    rhs = BlockVector(2)
    z = BlockVector(2)
    z.set_block(0, nlp.get_primals())
    z.set_block(1, nlp.get_duals())
    if output:
        print(f'{'Iter':<12}{'Objective':<12}{'Primal Infeasibility':<25}{'Dual Infeasibility':<25}{'Elapsed Time':<15}')
    for _iter in range(max_iter):
        nlp.set_primals(z.get_block(0))
        nlp.set_duals(z.get_block(1))
        grad_lag = nlp.evaluate_grad_objective() + nlp.evaluate_jacobian_eq().transpose() * z.get_block(1)
        residuals = nlp.evaluate_eq_constraints()
        if output:
            print(f'{_iter:<12}{nlp.evaluate_objective():<12.2e}{np.abs(residuals).max():<25.2e}{np.abs(grad_lag).max():<25.2e}{time.time() - t0:<15.2e}')
        if np.abs(grad_lag).max() <= tol and np.abs(residuals).max() <= tol:
            break
        kkt.set_block(0, 0, nlp.evaluate_hessian_lag())
        kkt.set_block(1, 0, nlp.evaluate_jacobian_eq())
        kkt.set_block(0, 1, nlp.evaluate_jacobian_eq().transpose())
        rhs.set_block(0, grad_lag)
        rhs.set_block(1, residuals)
        delta, res = linear_solver.solve(kkt, -rhs)
        assert res.status == LinearSolverStatus.successful
        z += delta