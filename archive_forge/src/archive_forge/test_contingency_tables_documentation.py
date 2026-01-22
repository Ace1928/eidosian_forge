import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm

    library(DescTools)
    data = array(c(313, 512, 19, 89,
                   207, 353, 8, 17,
                   205, 120, 391, 202,
                   278, 139, 244, 131,
                   138, 53, 299, 94,
                   351, 22, 317, 24),
                   dim=c(2, 2, 6))
    rslt = mantelhaen.test(data)
    bd1 = BreslowDayTest(data, correct=FALSE)
    bd2 = BreslowDayTest(data, correct=TRUE)
    