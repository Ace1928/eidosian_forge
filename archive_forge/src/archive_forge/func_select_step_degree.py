import numpy as np
import math
from mpl_toolkits.axisartist.grid_finder import ExtremeFinderSimple
def select_step_degree(dv):
    degree_limits_ = [1.5, 3, 7, 13, 20, 40, 70, 120, 270, 520]
    degree_steps_ = [1, 2, 5, 10, 15, 30, 45, 90, 180, 360]
    degree_factors = [1.0] * len(degree_steps_)
    minsec_limits_ = [1.5, 2.5, 3.5, 8, 11, 18, 25, 45]
    minsec_steps_ = [1, 2, 3, 5, 10, 15, 20, 30]
    minute_limits_ = np.array(minsec_limits_) / 60
    minute_factors = [60.0] * len(minute_limits_)
    second_limits_ = np.array(minsec_limits_) / 3600
    second_factors = [3600.0] * len(second_limits_)
    degree_limits = [*second_limits_, *minute_limits_, *degree_limits_]
    degree_steps = [*minsec_steps_, *minsec_steps_, *degree_steps_]
    degree_factors = [*second_factors, *minute_factors, *degree_factors]
    n = np.searchsorted(degree_limits, dv)
    step = degree_steps[n]
    factor = degree_factors[n]
    return (step, factor)