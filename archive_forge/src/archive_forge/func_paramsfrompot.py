import numpy as np
from scipy import stats
from scipy.special import comb
from scipy.stats.distributions import rv_continuous
import matplotlib.pyplot as plt
from numpy import where, inf
from numpy import abs as np_abs
def paramsfrompot(thresh, shape, scalepot):
    return (shape, scalepot + shape * thresh)