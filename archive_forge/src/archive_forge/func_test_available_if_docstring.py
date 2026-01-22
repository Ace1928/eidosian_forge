import pickle
import pytest
from sklearn.utils.metaestimators import available_if
def test_available_if_docstring():
    assert 'This is a mock available_if function' in str(AvailableParameterEstimator.__dict__['available_func'].__doc__)
    assert 'This is a mock available_if function' in str(AvailableParameterEstimator.available_func.__doc__)
    assert 'This is a mock available_if function' in str(AvailableParameterEstimator().available_func.__doc__)