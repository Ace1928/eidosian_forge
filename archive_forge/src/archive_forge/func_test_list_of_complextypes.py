import re
import unittest
from wsme import exc
from wsme import types
def test_list_of_complextypes(self):

    class A(object):
        bs = types.wsattr(['B'])

    class B(object):
        i = int
    types.register_type(A)
    types.register_type(B)
    assert A.bs.datatype.item_type is B