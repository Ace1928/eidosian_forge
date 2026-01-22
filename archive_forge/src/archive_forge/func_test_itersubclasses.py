import base64
import hashlib
import hmac
from unittest import mock
import uuid
from osprofiler import _utils as utils
from osprofiler.tests import test
def test_itersubclasses(self):

    class A(object):
        pass

    class B(A):
        pass

    class C(A):
        pass

    class D(C):
        pass
    self.assertEqual([B, C, D], list(utils.itersubclasses(A)))

    class E(type):
        pass
    self.assertEqual([], list(utils.itersubclasses(E)))