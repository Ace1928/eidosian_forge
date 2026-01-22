from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def scale_data(self):
    """
        Perform symmetric diagonal scaling via equilibration
        """
    n = self.work.data.n
    m = self.work.data.m
    s_temp = np.ones(n + m)
    c = 1.0
    P = self.work.data.P
    q = self.work.data.q
    A = self.work.data.A
    l = self.work.data.l
    u = self.work.data.u
    D = spspa.eye(n)
    if m == 0:
        E = spspa.csc_matrix((0, 0))
    else:
        E = spspa.eye(m)
    for i in range(self.work.settings.scaling):
        norm_cols = self._norm_KKT_cols(P, A)
        norm_cols = self._limit_scaling(norm_cols)
        sqrt_norm_cols = np.sqrt(norm_cols)
        s_temp = np.reciprocal(sqrt_norm_cols)
        D_temp = spspa.diags(s_temp[:self.work.data.n])
        if m == 0:
            E_temp = spspa.csc_matrix((0, 0))
        else:
            E_temp = spspa.diags(s_temp[self.work.data.n:])
        P = D_temp.dot(P.dot(D_temp)).tocsc()
        A = E_temp.dot(A.dot(D_temp)).tocsc()
        q = D_temp.dot(q)
        l = E_temp.dot(l)
        u = E_temp.dot(u)
        D = D_temp.dot(D)
        E = E_temp.dot(E)
        norm_P_cols = spla.norm(P, np.inf, axis=0).mean()
        inf_norm_q = np.linalg.norm(q, np.inf)
        inf_norm_q = self._limit_scaling(inf_norm_q)
        scale_cost = np.maximum(inf_norm_q, norm_P_cols)
        scale_cost = self._limit_scaling(scale_cost)
        scale_cost = 1.0 / scale_cost
        c_temp = scale_cost
        P = c_temp * P
        q = c_temp * q
        c = c_temp * c
    if self.work.settings.verbose:
        print('Final cost scaling = %.10f' % c)
    self.work.data = problem((n, m), P.data, P.indices, P.indptr, q, A.data, A.indices, A.indptr, l, u)
    self.work.scaling = scaling()
    self.work.scaling.D = D
    self.work.scaling.Dinv = spspa.diags(np.reciprocal(D.diagonal()))
    self.work.scaling.E = E
    if m == 0:
        self.work.scaling.Einv = E
    else:
        self.work.scaling.Einv = spspa.diags(np.reciprocal(E.diagonal()))
    self.work.scaling.c = c
    self.work.scaling.cinv = 1.0 / c