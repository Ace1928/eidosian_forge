import numpy as np
empirical mean residual life or expected shortfall

    Parameters
    ----------
    x : 1-dimensional array_like
    frac : list[float], optional
        All entries must be between 0 and 1
    alpha : float, default 0.05
        FIXME: not actually used.

    TODO:
        check formula for std of mean
        does not include case for all observations
        last observations std is zero
        vectorize loop using cumsum
        frac does not work yet
    