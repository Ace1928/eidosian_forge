import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('Encoder', [OneHotEncoder, OrdinalEncoder])
def test_encoder_not_fitted(Encoder):
    """Check that we raise a `NotFittedError` by calling transform before fit with
    the encoders.

    One could expect that the passing the `categories` argument to the encoder
    would make it stateless. However, `fit` is making a couple of check, such as the
    position of `np.nan`.
    """
    X = np.array([['A'], ['B'], ['C']], dtype=object)
    encoder = Encoder(categories=[['A', 'B', 'C']])
    with pytest.raises(NotFittedError):
        encoder.transform(X)