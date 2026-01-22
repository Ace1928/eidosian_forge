import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def test_to_uniqueify_list():
    assert uniqueify_list([1, 2, 3]) == [1, 2, 3]
    assert uniqueify_list([1, 3, 3, 2, 3, 1]) == [1, 3, 2]
    assert uniqueify_list([3, 2, 1, 4, 1, 2, 3]) == [3, 2, 1, 4]