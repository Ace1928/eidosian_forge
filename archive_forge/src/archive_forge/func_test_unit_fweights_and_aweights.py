import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
def test_unit_fweights_and_aweights(self):
    assert_allclose(cov(self.x2, fweights=self.frequencies, aweights=self.unit_weights), cov(self.x2_repeats))
    assert_allclose(cov(self.x1, fweights=self.frequencies, aweights=self.unit_weights), self.res2)
    assert_allclose(cov(self.x1, fweights=self.unit_frequencies, aweights=self.unit_weights), self.res1)
    assert_allclose(cov(self.x1, fweights=self.unit_frequencies, aweights=self.weights), self.res3)
    assert_allclose(cov(self.x1, fweights=self.unit_frequencies, aweights=3.0 * self.weights), cov(self.x1, aweights=self.weights))
    assert_allclose(cov(self.x1, fweights=self.unit_frequencies, aweights=self.unit_weights), self.res1)