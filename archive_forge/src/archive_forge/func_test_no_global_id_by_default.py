import fixtures
import hashlib
import uuid
import warnings
from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
def test_no_global_id_by_default(self):
    ctx = context.RequestContext()
    self.assertIsNone(ctx.global_request_id)
    d = ctx.to_dict()
    self.assertIsNone(d['global_request_id'])