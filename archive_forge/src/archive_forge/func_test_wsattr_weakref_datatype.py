import re
import unittest
from wsme import exc
from wsme import types
def test_wsattr_weakref_datatype(self):
    import weakref
    a = types.wsattr(int)
    a.datatype = weakref.ref(int)
    assert a.datatype is int