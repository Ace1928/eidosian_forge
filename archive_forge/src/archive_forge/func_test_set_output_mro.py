import importlib
from collections import namedtuple
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn._config import config_context, get_config
from sklearn.preprocessing import StandardScaler
from sklearn.utils._set_output import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_set_output_mro():
    """Check that multi-inheritance resolves to the correct class method.

    Non-regression test gh-25293.
    """

    class Base(_SetOutputMixin):

        def transform(self, X):
            return 'Base'

    class A(Base):
        pass

    class B(Base):

        def transform(self, X):
            return 'B'

    class C(A, B):
        pass
    assert C().transform(None) == 'B'