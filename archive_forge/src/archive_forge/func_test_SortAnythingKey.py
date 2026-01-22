import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def test_SortAnythingKey():
    assert sorted([20, 10, 0, 15], key=SortAnythingKey) == [0, 10, 15, 20]
    assert sorted([10, -1.5], key=SortAnythingKey) == [-1.5, 10]
    assert sorted([10, 'a', 20.5, 'b'], key=SortAnythingKey) == [10, 20.5, 'a', 'b']

    class a(object):
        pass

    class b(object):
        pass

    class z(object):
        pass
    a_obj = a()
    b_obj = b()
    z_obj = z()
    o_obj = object()
    assert sorted([z_obj, a_obj, 1, b_obj, o_obj], key=SortAnythingKey) == [1, a_obj, b_obj, o_obj, z_obj]