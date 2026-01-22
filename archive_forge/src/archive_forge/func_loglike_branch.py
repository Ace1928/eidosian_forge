import numpy as np
import numpy.lib.recfunctions as recf
from scipy import optimize
def loglike_branch(self, params, tau):
    ivs = []
    for b in branches:
        probs, iv = self.loglike_leafbranch(params, tau)
        ivs.append(iv)
    ivs = np.column_stack(ivs)
    exptiv = np.exp(tau * ivs)
    sumexptiv = exptiv.sum(1)
    logsumexpxb = np.log(sumexpxb)
    probs = exptiv / sumexptiv[:, None]