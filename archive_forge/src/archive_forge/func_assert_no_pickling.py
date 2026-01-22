import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def assert_no_pickling(obj):
    import pickle
    import pytest
    pytest.raises(NotImplementedError, pickle.dumps, obj)