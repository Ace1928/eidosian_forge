import unittest
import numpy as np
from statsmodels.multivariate.factor_rotation._wrappers import rotate_factors
from statsmodels.multivariate.factor_rotation._gpa_rotation import (
from statsmodels.multivariate.factor_rotation._analytic_rotation import (
def get_A(self):
    return self.str2matrix('\n         .830 -.396\n         .818 -.469\n         .777 -.470\n         .798 -.401\n         .786  .500\n         .672  .458\n         .594  .444\n         .647  .333\n        ')