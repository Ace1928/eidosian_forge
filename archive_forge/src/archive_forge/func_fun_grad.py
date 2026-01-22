from numpy.testing import assert_, assert_equal
import pytest
from pytest import raises as assert_raises, warns as assert_warns
import numpy as np
from scipy.optimize import root
def fun_grad(x, ignored):
    return (fun(x, ignored), grad(x, ignored))