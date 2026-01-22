import unittest
import numpy as np
from statsmodels.multivariate.factor_rotation._wrappers import rotate_factors
from statsmodels.multivariate.factor_rotation._gpa_rotation import (
from statsmodels.multivariate.factor_rotation._analytic_rotation import (
def get_H(self):
    return self.str2matrix('\n          .8 -.3\n          .8 -.4\n          .7 -.4\n          .9 -.4\n          .8  .5\n          .6  .4\n          .5  .4\n          .6  .3\n        ')