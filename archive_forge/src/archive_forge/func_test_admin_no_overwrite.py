import fixtures
import hashlib
import uuid
import warnings
from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
def test_admin_no_overwrite(self):
    ctx1 = context.RequestContext(overwrite=True)
    context.get_admin_context()
    self.assertIs(context.get_current(), ctx1)
    self.assertFalse(ctx1.is_admin)