import numpy as np
from scipy import stats
def predicted(self):
    """Value of the function evaluated at the attached params.

        Note: This is not equal to the expected value if the transformation is
        nonlinear. If params is the maximum likelihood estimate, then
        `predicted` is the maximum likelihood estimate of the value of the
        nonlinear function.
        """
    predicted = self.fun(self.params)
    if predicted.ndim > 1:
        predicted = predicted.squeeze()
    return predicted