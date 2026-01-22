
Critical value polynomials and related quantities for the bounds test of

Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). Bounds testing approaches
   to the analysis of level relationships. Journal of applied econometrics,
   16(3), 289-326.

These were computed using 32,000,000 simulations for each key using the
methodology of PSS, who only used 40,000. The asymptotic P-value response
functions were computed based on the simulated value. Critical values
are the point estimates for the respective quantiles. The simulation code
is contained in pss.py. The output files from this function are then
transformed using pss-process.py.

The format of the keys are (k, case, I1) where

* k is is the number of x variables included in the model (0 is an ADF)
* case is 1, 2, 3, 4 or 5 and corresponds to the PSS paper
* I1 is True if X contains I1 variables and False if X is stationary

The parameters are for polynomials of order 3 (large) or 2 (small).
stat_star is the value where the switch between large and small occurs.
Stat values less then stat_star use large_p, while values above use
small_p. In all cases the stat is logged prior to computing the p-value
so that the p-value is

1 - Phi(c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3)

where x = np.log(stat) and Phi() is the normal cdf.

When this the models, the polynomial is evaluated at the natural log of the
test statistic and then the normal CDF of this value is computed to produce
the p-value.
