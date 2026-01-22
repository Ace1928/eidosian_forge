import pickle
import pytest
from sklearn.utils.deprecation import _is_deprecated, deprecated
@deprecated('n_features_ is deprecated')
@property
def n_features_(self):
    """Number of input features."""
    return 10