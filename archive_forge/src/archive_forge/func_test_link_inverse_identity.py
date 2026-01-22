import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn._loss.link import (
@pytest.mark.parametrize('link', LINK_FUNCTIONS)
def test_link_inverse_identity(link, global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    link = link()
    n_samples, n_classes = (100, None)
    if link.is_multiclass:
        n_classes = 10
        raw_prediction = rng.uniform(low=-20, high=20, size=(n_samples, n_classes))
        if isinstance(link, MultinomialLogit):
            raw_prediction = link.symmetrize_raw_prediction(raw_prediction)
    elif isinstance(link, HalfLogitLink):
        raw_prediction = rng.uniform(low=-10, high=10, size=n_samples)
    else:
        raw_prediction = rng.uniform(low=-20, high=20, size=n_samples)
    assert_allclose(link.link(link.inverse(raw_prediction)), raw_prediction)
    y_pred = link.inverse(raw_prediction)
    assert_allclose(link.inverse(link.link(y_pred)), y_pred)