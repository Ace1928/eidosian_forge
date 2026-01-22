from abc import ABCMeta, abstractmethod
from pyomo.contrib.pynumero.interfaces import pyomo_nlp, ampl_nlp
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
import numpy as np
import scipy.sparse
from pyomo.common.timing import HierarchicalTimer
def evaluate_primal_dual_kkt_rhs(self, timer=None):
    if timer is None:
        timer = HierarchicalTimer()
    timer.start('eval grad obj')
    grad_obj = self.get_obj_factor() * self.evaluate_grad_objective()
    timer.stop('eval grad obj')
    timer.start('eval jac')
    jac_eq = self._nlp.evaluate_jacobian_eq()
    jac_ineq = self._nlp.evaluate_jacobian_ineq()
    timer.stop('eval jac')
    timer.start('eval cons')
    eq_resid = self._nlp.evaluate_eq_constraints()
    ineq_resid = self._nlp.evaluate_ineq_constraints() - self._slacks
    timer.stop('eval cons')
    timer.start('grad_lag_primals')
    grad_lag_primals = grad_obj + jac_eq.transpose() * self._nlp.get_duals_eq() + jac_ineq.transpose() * self._nlp.get_duals_ineq() - self._barrier / (self._nlp.get_primals() - self._nlp.primals_lb()) + self._barrier / (self._nlp.primals_ub() - self._nlp.get_primals())
    timer.stop('grad_lag_primals')
    timer.start('grad_lag_slacks')
    grad_lag_slacks = -self._nlp.get_duals_ineq() - self._barrier / (self._slacks - self._nlp.ineq_lb()) + self._barrier / (self._nlp.ineq_ub() - self._slacks)
    timer.stop('grad_lag_slacks')
    rhs = BlockVector(4)
    rhs.set_block(0, grad_lag_primals)
    rhs.set_block(1, grad_lag_slacks)
    rhs.set_block(2, eq_resid)
    rhs.set_block(3, ineq_resid)
    rhs = -rhs
    return rhs