import pickle
import pytest
from sklearn.utils.metaestimators import available_if
def test_available_if():
    assert hasattr(AvailableParameterEstimator(), 'available_func')
    assert not hasattr(AvailableParameterEstimator(available=False), 'available_func')