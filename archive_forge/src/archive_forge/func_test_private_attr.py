import re
import unittest
from wsme import exc
from wsme import types
def test_private_attr(self):

    class WithPrivateAttrs(object):
        _private = 12
    types.register_type(WithPrivateAttrs)
    assert len(WithPrivateAttrs._wsme_attributes) == 0