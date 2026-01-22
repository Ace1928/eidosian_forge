import fixtures
import hashlib
import uuid
import warnings
from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
def test_from_dict_extended(self):
    initial = TestContext(auth_token_info='foo')
    dct = initial.to_dict()
    final = TestContext.from_dict(dct)
    self.assertEqual('foo', final.auth_token_info)
    self.assertEqual(dct, final.to_dict())