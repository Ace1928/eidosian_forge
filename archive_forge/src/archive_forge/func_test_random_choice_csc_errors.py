import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_array_almost_equal
from scipy.special import comb
from sklearn.utils._random import _our_rand_r_py
from sklearn.utils.random import _random_choice_csc, sample_without_replacement
def test_random_choice_csc_errors():
    classes = [np.array([0, 1]), np.array([0, 1, 2, 3])]
    class_probabilities = [np.array([0.5, 0.5]), np.array([0.6, 0.1, 0.3])]
    with pytest.raises(ValueError):
        _random_choice_csc(4, classes, class_probabilities, 1)
    classes = [np.array(['a', '1']), np.array(['z', '1', '2'])]
    class_probabilities = [np.array([0.5, 0.5]), np.array([0.6, 0.1, 0.3])]
    with pytest.raises(ValueError):
        _random_choice_csc(4, classes, class_probabilities, 1)
    classes = [np.array([4.2, 0.1]), np.array([0.1, 0.2, 9.4])]
    class_probabilities = [np.array([0.5, 0.5]), np.array([0.6, 0.1, 0.3])]
    with pytest.raises(ValueError):
        _random_choice_csc(4, classes, class_probabilities, 1)
    classes = [np.array([0, 1]), np.array([0, 1, 2])]
    class_probabilities = [np.array([0.5, 0.6]), np.array([0.6, 0.1, 0.3])]
    with pytest.raises(ValueError):
        _random_choice_csc(4, classes, class_probabilities, 1)