from statsmodels.compat.numpy import NP_LT_2
import numpy as np
import pandas as pd
def is_design_matrix(obj):
    from patsy import DesignMatrix
    return isinstance(obj, DesignMatrix)