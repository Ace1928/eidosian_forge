import fixtures
import hashlib
import uuid
import warnings
from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
def test_dict_empty_user_identity(self):
    ctx = context.RequestContext()
    d = ctx.to_dict()
    self.assertEqual('- - - - -', d['user_identity'])