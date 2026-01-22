from __future__ import print_function, with_statement
import logging
import os
import sys
import time
import bz2
import itertools
import numpy as np
import scipy.linalg
import gensim
def print_error(name, aat, u, s, ideal_nf, ideal_n2):
    err = -np.dot(u, np.dot(np.diag(s), u.T))
    err += aat
    nf, n2 = (np.linalg.norm(err), norm2(err))
    print('%s error: norm_frobenius=%f (/ideal=%g), norm2=%f (/ideal=%g), RMSE=%g' % (name, nf, nf / ideal_nf, n2, n2 / ideal_n2, rmse(err)))
    sys.stdout.flush()