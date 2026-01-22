import numpy as np
from statsmodels.stats._knockoff import RegressionFDR

            Negative log-likelihood of z-scores.

            The function has three arguments, packed into a vector:

            mean : location parameter
            logscale : log of the scale parameter
            logitprop : logit of the proportion of true nulls

            The implementation follows section 4 from Efron 2008.
            