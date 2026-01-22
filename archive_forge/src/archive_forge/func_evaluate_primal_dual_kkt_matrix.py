from abc import ABCMeta, abstractmethod
from pyomo.contrib.pynumero.interfaces import pyomo_nlp, ampl_nlp
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
import numpy as np
import scipy.sparse
from pyomo.common.timing import HierarchicalTimer
def evaluate_primal_dual_kkt_matrix(self, timer=None):
    if timer is None:
        timer = HierarchicalTimer()
    timer.start('eval hess')
    hessian = self._nlp.evaluate_hessian_lag()
    timer.stop('eval hess')
    timer.start('eval jac')
    jac_eq = self._nlp.evaluate_jacobian_eq()
    jac_ineq = self._nlp.evaluate_jacobian_ineq()
    timer.stop('eval jac')
    duals_primals_lb = self._duals_primals_lb
    duals_primals_ub = self._duals_primals_ub
    duals_slacks_lb = self._duals_slacks_lb
    duals_slacks_ub = self._duals_slacks_ub
    primals = self._nlp.get_primals()
    timer.start('hess block')
    data = duals_primals_lb / (primals - self._nlp.primals_lb()) + duals_primals_ub / (self._nlp.primals_ub() - primals)
    n = self._nlp.n_primals()
    indices = np.arange(n)
    hess_block = scipy.sparse.coo_matrix((data, (indices, indices)), shape=(n, n))
    hess_block += hessian
    timer.stop('hess block')
    timer.start('slack block')
    data = duals_slacks_lb / (self._slacks - self._nlp.ineq_lb()) + duals_slacks_ub / (self._nlp.ineq_ub() - self._slacks)
    n = self._nlp.n_ineq_constraints()
    indices = np.arange(n)
    slack_block = scipy.sparse.coo_matrix((data, (indices, indices)), shape=(n, n))
    timer.stop('slack block')
    timer.start('set block')
    kkt = BlockMatrix(4, 4)
    kkt.set_block(0, 0, hess_block)
    kkt.set_block(1, 1, slack_block)
    kkt.set_block(2, 0, jac_eq)
    kkt.set_block(0, 2, jac_eq.transpose())
    kkt.set_block(3, 0, jac_ineq)
    kkt.set_block(0, 3, jac_ineq.transpose())
    kkt.set_block(3, 1, -scipy.sparse.identity(self._nlp.n_ineq_constraints(), format='coo'))
    kkt.set_block(1, 3, -scipy.sparse.identity(self._nlp.n_ineq_constraints(), format='coo'))
    timer.stop('set block')
    return kkt