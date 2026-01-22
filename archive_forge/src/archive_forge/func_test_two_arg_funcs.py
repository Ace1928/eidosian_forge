import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_two_arg_funcs(self):
    funcs = (random.uniform, random.normal, random.beta, random.gamma, random.f, random.noncentral_chisquare, random.vonmises, random.laplace, random.gumbel, random.logistic, random.lognormal, random.wald, random.binomial, random.negative_binomial)
    probfuncs = (random.binomial, random.negative_binomial)
    for func in funcs:
        if func in probfuncs:
            argTwo = np.array([0.5])
        else:
            argTwo = self.argTwo
        out = func(self.argOne, argTwo)
        assert_equal(out.shape, self.tgtShape)
        out = func(self.argOne[0], argTwo)
        assert_equal(out.shape, self.tgtShape)
        out = func(self.argOne, argTwo[0])
        assert_equal(out.shape, self.tgtShape)