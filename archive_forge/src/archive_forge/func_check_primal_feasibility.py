import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def check_primal_feasibility(self, places) -> None:
    all_cons = [c for c in self.constraints]
    for x in self.prob.variables():
        attrs = x.attributes
        if attrs['nonneg'] or attrs['pos']:
            all_cons.append(x >= 0)
        elif attrs['nonpos'] or attrs['neg']:
            all_cons.append(x <= 0)
        elif attrs['imag']:
            all_cons.append(x + cp.conj(x) == 0)
        elif attrs['symmetric']:
            all_cons.append(x == x.T)
        elif attrs['diag']:
            all_cons.append(x - cp.diag(cp.diag(x)) == 0)
        elif attrs['PSD']:
            all_cons.append(x >> 0)
        elif attrs['NSD']:
            all_cons.append(x << 0)
        elif attrs['hermitian']:
            all_cons.append(x == cp.conj(x.T))
        elif attrs['boolean'] or attrs['integer']:
            round_val = np.round(x.value)
            all_cons.append(x == round_val)
    for con in all_cons:
        viol = con.violation()
        if isinstance(viol, np.ndarray):
            viol = np.linalg.norm(viol, ord=2)
        self.tester.assertAlmostEqual(viol, 0, places)