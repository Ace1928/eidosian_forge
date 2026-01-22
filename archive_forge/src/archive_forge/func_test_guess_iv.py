import os
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy import stats
from scipy.optimize import differential_evolution
from .test_continuous_basic import distcont
from scipy.stats._distn_infrastructure import FitError
from scipy.stats._distr_params import distdiscrete
from scipy.stats import goodness_of_fit
def test_guess_iv(self):
    message = 'Guesses provided for the following unrecognized...'
    guess = {'n': 1, 'p': 0.5, '1': 255}
    with pytest.warns(RuntimeWarning, match=message):
        stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)
    message = 'Each element of `guess` must be a scalar...'
    guess = {'n': 1, 'p': 'hi'}
    with pytest.raises(ValueError, match=message):
        stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)
    guess = [1, 'f']
    with pytest.raises(ValueError, match=message):
        stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)
    guess = [[1, 2]]
    with pytest.raises(ValueError, match=message):
        stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)
    message = 'A `guess` sequence must contain at least 2...'
    guess = [1]
    with pytest.raises(ValueError, match=message):
        stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)
    message = 'A `guess` sequence may not contain more than 3...'
    guess = [1, 2, 3, 4]
    with pytest.raises(ValueError, match=message):
        stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)
    message = 'Guess for parameter `n` rounded.*|Guess for parameter `p` clipped.*'
    guess = {'n': 4.5, 'p': -0.5}
    with pytest.warns(RuntimeWarning, match=message):
        stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)
    message = 'Guess for parameter `loc` rounded...'
    guess = [5, 0.5, 0.5]
    with pytest.warns(RuntimeWarning, match=message):
        stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)
    message = 'Guess for parameter `p` clipped...'
    guess = {'n': 5, 'p': -0.5}
    with pytest.warns(RuntimeWarning, match=message):
        stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)
    message = 'Guess for parameter `loc` clipped...'
    guess = [5, 0.5, 1]
    with pytest.warns(RuntimeWarning, match=message):
        stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)