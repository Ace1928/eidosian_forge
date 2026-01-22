import numpy as np
import numpy.linalg as L
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly
def compute_P(self, Sinv):
    """projection matrix (nobs_i, nobs_i) (M in regression ?)  (JP check, guessing)
        Display (3.10) from Laird, Lange, Stram (see help(Unit))

        W - W X Sinv X' W'
        """
    t = np.dot(self.W, self.X)
    self.P = self.W - np.dot(np.dot(t, Sinv), t.T)