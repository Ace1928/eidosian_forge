import numpy as np
from scipy import special, stats
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.tsa.arma_mle import Arma

        Loglikelihood for arma model for each observation, t-distribute

        Notes
        -----
        The ancillary parameter is assumed to be the last element of
        the params vector
        