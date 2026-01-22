import pickle
import os
import tempfile
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (sarimax, structural, varmax,
from numpy.testing import assert_allclose
def test_existing_pickle():
    pkl_file = os.path.join(current_path, 'results', 'sm-0.9-sarimax.pkl')
    loaded = sarimax.SARIMAXResults.load(pkl_file)
    assert isinstance(loaded, sarimax.SARIMAXResultsWrapper)