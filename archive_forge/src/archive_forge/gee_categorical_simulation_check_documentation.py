from statsmodels.compat.python import lrange
import numpy as np
from scipy import stats
from statsmodels.genmod.generalized_estimating_equations import GEE,\
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import GlobalOddsRatio
from .gee_gaussian_simulation_check import GEE_simulator

Assesment of Generalized Estimating Equations using simulation.

This script checks ordinal and nominal models for multinomial data.

See the generated file "gee_categorical_simulation_check.txt" for
results.
