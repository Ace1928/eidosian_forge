import time
import numpy as np
from numpy.linalg import eigh
from ase.optimize.optimize import Optimizer
def update_hessian_bofill(self, pos, G):
    print('update Bofill')
    n = len(self.hessian)
    dgrad = G - self.oldG
    dpos = pos - self.oldpos
    absdpos = np.dot(dpos, dpos)
    if absdpos < self.eps:
        return
    dotg = np.dot(dgrad, dpos)
    tvec = dgrad - np.dot(dpos, self.hessian)
    tvecdot = np.dot(tvec, tvec)
    tvecdpos = np.dot(tvec, dpos)
    coef1 = 1.0 - tvecdpos * tvecdpos / (absdpos * tvecdot)
    coef2 = (1.0 - coef1) * absdpos / tvecdpos
    coef3 = coef1 * tvecdpos / absdpos
    dott = np.dot(dpos, tvec)
    if abs(dott) > self.eps and abs(dotg) > self.eps:
        for i in range(n):
            for j in range(n):
                h = coef1 * (tvec[i] * dpos[j] + dpos[i] * tvec[j]) - dpos[i] * dpos[j] * coef3 + coef2 * tvec[i] * tvec[j]
                h *= 1.0 / absdpos
                self.hessian[i][j] += h