from functools import partial
import numpy as np
from sklearn.base import (
from sklearn.metrics._scorer import _Scorer, mean_squared_error
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.utils._metadata_requests import (
from sklearn.utils.metadata_routing import (
from sklearn.utils.multiclass import _check_partial_fit_first_call
def check_recorded_metadata(obj, method, split_params=tuple(), **kwargs):
    """Check whether the expected metadata is passed to the object's method.

    Parameters
    ----------
    obj : estimator object
        sub-estimator to check routed params for
    method : str
        sub-estimator's method where metadata is routed to
    split_params : tuple, default=empty
        specifies any parameters which are to be checked as being a subset
        of the original values.
    """
    records = getattr(obj, '_records', dict()).get(method, dict())
    assert set(kwargs.keys()) == set(records.keys())
    for key, value in kwargs.items():
        recorded_value = records[key]
        if key in split_params and recorded_value is not None:
            assert np.isin(recorded_value, value).all()
        else:
            assert recorded_value is value