import numpy as np
import numpy.lib.recfunctions
from statsmodels.compat.python import lmap
from statsmodels.regression.linear_model import OLS
drop names from a list of strings,
    names to drop are in space delimited list
    does not change original list
    