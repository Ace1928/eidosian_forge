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
def rmse(diff):
    return np.sqrt(1.0 * np.multiply(diff, diff).sum() / diff.size)