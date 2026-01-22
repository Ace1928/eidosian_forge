import re
import unittest
from wsme import exc
from wsme import types
def test_inspect_with_property(self):

    class AType(object):

        @property
        def test(self):
            return 'test'
    types.register_type(AType)
    assert len(AType._wsme_attributes) == 0
    assert AType().test == 'test'