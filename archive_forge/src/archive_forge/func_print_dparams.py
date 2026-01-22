from statsmodels.compat.python import lrange
import scipy
import numpy as np
from itertools import product
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Autoregressive, Nested
def print_dparams(self, dparams_est):
    for j in range(len(self.nest_sizes)):
        OUT.write('Nest %d variance estimate:  %8.4f\n' % (j + 1, dparams_est[j]))
        OUT.write('Nest %d variance truth:     %8.4f\n' % (j + 1, self.dparams[j]))
    OUT.write('Error variance estimate:   %8.4f\n' % (dparams_est[-1] - sum(dparams_est[0:-1])))
    OUT.write('Error variance truth:      %8.4f\n' % self.error_sd ** 2)
    OUT.write('\n')