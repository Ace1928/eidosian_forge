import numpy as np
from scipy import special
from scipy.optimize import OptimizeResult
from scipy.optimize._zeros_py import (  # noqa: F401
def post_func_eval(x, fj, work):
    if work.log:
        fj[work.abinf] += np.log(1 + work.xj[work.abinf] ** 2) - 2 * np.log(1 - work.xj[work.abinf] ** 2)
        fj[work.binf] -= 2 * np.log(work.xj[work.binf])
    else:
        fj[work.abinf] *= (1 + work.xj[work.abinf] ** 2) / (1 - work.xj[work.abinf] ** 2) ** 2
        fj[work.binf] *= work.xj[work.binf] ** (-2.0)
    fjwj, Sn = _euler_maclaurin_sum(fj, work)
    if work.Sk.shape[-1]:
        Snm1 = work.Sk[:, -1]
        Sn = special.logsumexp([Snm1 - np.log(2), Sn], axis=0) if log else Snm1 / 2 + Sn
    work.fjwj = fjwj
    work.Sn = Sn