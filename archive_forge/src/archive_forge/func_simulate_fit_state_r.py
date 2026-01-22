from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.pytest import pytest_warns
import os
import re
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.holtwinters import (
from statsmodels.tsa.holtwinters._exponential_smoothers import (
from statsmodels.tsa.holtwinters._smoothers import (
@pytest.fixture
def simulate_fit_state_r():
    '''
    The final state from the R model fits to get an exact comparison
    Obtained with this R script:

    library(magrittr)
    library(fpp2)
    library(forecast)

    concat <- function(...) {
      return(paste(..., sep=""))
    }

    as_dict_string <- function(named) {
      string <- '{'
      for (name in names(named)) {
        string <- concat(string, """, name, "": ", named[name], ", ")
      }
      string <- concat(string, '}')
      return(string)
    }

    get_var <- function(named, name) {
      if (name %in% names(named))
        val <- c(named[name])
      else
        val <- c(NaN)
      names(val) <- c(name)
      return(val)
    }

    error <- c("A", "M")
    trend <- c("A", "M", "N")
    seasonal <- c("A", "M", "N")
    models <- outer(error, trend, FUN = "concat") %>%
      outer(seasonal, FUN = "concat") %>% as.vector

    # innov from np.random.seed(0); np.random.randn(4)
    innov <- c(1.76405235, 0.40015721, 0.97873798, 2.2408932)
    n <- length(austourists) + 1

    # print fit parameters and final states
    for (damped in c(TRUE, FALSE)) {
      print(paste("damped =", damped))
      for (model in models) {
        state <- tryCatch((function(){
          fit <- ets(austourists, model = model, damped = damped)
          pars <- c()
          # alpha, beta, gamma, phi
          for (name in c("alpha", "beta", "gamma", "phi")) {
            pars <- c(pars, get_var(fit$par, name))
          }
          # l, b, s1, s2, s3, s4
          states <- c()
          for (name in c("l", "b", "s1", "s2", "s3", "s4"))
            states <- c(states, get_var(fit$states[n,], name))
          c(pars, states)
        })(),
        error = function(e) rep(NA, 10))
        cat(concat(""", model, "": ", as_dict_string(state), ",
"))
      }
    }
    '''
    damped = {'AAA': {'alpha': 0.35445427317618, 'beta': 0.0320074905894167, 'gamma': 0.399933869627979, 'phi': 0.979999965983533, 'l': 62.003405788717, 'b': 0.706524957599738, 's1': 3.58786406600866, 's2': -0.0747450283892903, 's3': -11.7569356589817, 's4': 13.3818805055271}, 'MAA': {'alpha': 0.31114284033284, 'beta': 0.0472138763848083, 'gamma': 0.309502324693322, 'phi': 0.870889202791893, 'l': 59.2902342851514, 'b': 0.62538315801909, 's1': 5.66660224738038, 's2': 2.16097311633352, 's3': -9.20020909069337, 's4': 15.3505801601698}, 'MAM': {'alpha': 0.483975835390643, 'beta': 0.00351728130401287, 'gamma': 0.00011309784353818, 'phi': 0.979999998322032, 'l': 63.0042707536293, 'b': 0.275035160634846, 's1': 1.03531670491486, 's2': 0.960515682506077, 's3': 0.770086097577864, 's4': 1.23412213281709}, 'MMM': {'alpha': 0.523526123191035, 'beta': 0.000100021136675999, 'gamma': 0.000100013723372502, 'phi': 0.971025672907157, 'l': 63.2030316675533, 'b': 1.00458391644788, 's1': 1.03476354353096, 's2': 0.959953222294316, 's3': 0.771346403552048, 's4': 1.23394845160922}, 'AAN': {'alpha': 0.014932817259302, 'beta': 0.0149327068053362, 'gamma': np.nan, 'phi': 0.979919958387887, 'l': 60.0651024395378, 'b': 0.699112782133822, 's1': np.nan, 's2': np.nan, 's3': np.nan, 's4': np.nan}, 'MAN': {'alpha': 0.0144217343786778, 'beta': 0.0144216994589862, 'gamma': np.nan, 'phi': 0.979999719878659, 'l': 60.1870032363649, 'b': 0.698421913047609, 's1': np.nan, 's2': np.nan, 's3': np.nan, 's4': np.nan}, 'MMN': {'alpha': 0.015489181776072, 'beta': 0.0154891632646377, 'gamma': np.nan, 'phi': 0.975139118496093, 'l': 60.1855946424729, 'b': 1.00999589024928, 's1': np.nan, 's2': np.nan, 's3': np.nan, 's4': np.nan}}
    undamped = {'AAA': {'alpha': 0.20281951627363, 'beta': 0.000169786227368617, 'gamma': 0.464523797585052, 'phi': np.nan, 'l': 62.5598121416791, 'b': 0.578091734736357, 's1': 2.61176734723357, 's2': -1.24386240029203, 's3': -12.9575427049515, 's4': 12.2066400808086}, 'MAA': {'alpha': 0.416371920801538, 'beta': 0.000100008012920072, 'gamma': 0.352943901103959, 'phi': np.nan, 'l': 62.0497742976079, 'b': 0.450130087198346, 's1': 3.50368220490457, 's2': -0.0544297321113539, 's3': -11.6971093199679, 's4': 13.1974985095916}, 'ANA': {'alpha': 0.54216694759434, 'beta': np.nan, 'gamma': 0.392030170511872, 'phi': np.nan, 'l': 57.606831186929, 'b': np.nan, 's1': 8.29613785790501, 's2': 4.6033791939889, 's3': -7.43956343440823, 's4': 17.722316385643}, 'MNA': {'alpha': 0.532842556756286, 'beta': np.nan, 'gamma': 0.346387433608713, 'phi': np.nan, 'l': 58.0372808528325, 'b': np.nan, 's1': 7.70802088750111, 's2': 4.14885814748503, 's3': -7.72115936226225, 's4': 17.1674660340923}, 'MAM': {'alpha': 0.315621390571192, 'beta': 0.000100011993615961, 'gamma': 0.000100051297784532, 'phi': np.nan, 'l': 62.4082004238551, 'b': 0.513327867101983, 's1': 1.03713425342421, 's2': 0.959607104686072, 's3': 0.770172817592091, 's4': 1.23309264451638}, 'MMM': {'alpha': 0.546068965886, 'beta': 0.0737816453485457, 'gamma': 0.000100031693302807, 'phi': np.nan, 'l': 63.8203866275649, 'b': 1.01833305374778, 's1': 1.03725227137871, 's2': 0.961177239042923, 's3': 0.771173487523454, 's4': 1.23036313932852}, 'MNM': {'alpha': 0.608993139624813, 'beta': np.nan, 'gamma': 0.000167258612971303, 'phi': np.nan, 'l': 63.1472153330648, 'b': np.nan, 's1': 1.0384840572776, 's2': 0.961456755855531, 's3': 0.768427399477366, 's4': 1.23185085956321}, 'AAN': {'alpha': 0.0097430554119077, 'beta': 0.00974302759255084, 'gamma': np.nan, 'phi': np.nan, 'l': 61.1430969243248, 'b': 0.759041621012503, 's1': np.nan, 's2': np.nan, 's3': np.nan, 's4': np.nan}, 'MAN': {'alpha': 0.0101749952821338, 'beta': 0.0101749138539332, 'gamma': np.nan, 'phi': np.nan, 'l': 61.6020426238699, 'b': 0.761407500773051, 's1': np.nan, 's2': np.nan, 's3': np.nan, 's4': np.nan}, 'MMN': {'alpha': 0.0664382968951546, 'beta': 0.000100001678373356, 'gamma': np.nan, 'phi': np.nan, 'l': 60.7206911970871, 'b': 1.01221899136391, 's1': np.nan, 's2': np.nan, 's3': np.nan, 's4': np.nan}, 'ANN': {'alpha': 0.196432515825523, 'beta': np.nan, 'gamma': np.nan, 'phi': np.nan, 'l': 58.7718395431632, 'b': np.nan, 's1': np.nan, 's2': np.nan, 's3': np.nan, 's4': np.nan}, 'MNN': {'alpha': 0.205985314333856, 'beta': np.nan, 'gamma': np.nan, 'phi': np.nan, 'l': 58.9770839944419, 'b': np.nan, 's1': np.nan, 's2': np.nan, 's3': np.nan, 's4': np.nan}}
    return {True: damped, False: undamped}