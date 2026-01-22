import unittest
import numpy as np
from statsmodels.multivariate.factor_rotation._wrappers import rotate_factors
from statsmodels.multivariate.factor_rotation._gpa_rotation import (
from statsmodels.multivariate.factor_rotation._analytic_rotation import (
def get_W(self):
    return self.str2matrix('\n        1 0\n        0 1\n        0 0\n        1 1\n        1 0\n        1 0\n        0 1\n        1 0\n        ')