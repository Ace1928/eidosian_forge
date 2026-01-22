import os
import numpy as np
from statsmodels.duration.survfunc import (
from numpy.testing import assert_allclose
import pandas as pd
import pytest

library(survival)
ti1 = c(3, 1, 2, 3, 2, 1, 5, 3)
st1 = c(0, 1, 1, 1, 0, 0, 1, 0)
ti2 = c(1, 1, 2, 3, 7, 1, 5, 3, 9)
st2 = c(0, 1, 0, 0, 1, 0, 1, 0, 1)

ti = c(ti1, ti2)
st = c(st1, st2)
ix = c(rep(1, length(ti1)), rep(2, length(ti2)))
sd = survdiff(Surv(ti, st) ~ ix)
