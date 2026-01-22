import fixtures
import hashlib
import uuid
import warnings
from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
def test_no_overwrite(self):
    ctx1 = context.RequestContext(overwrite=True)
    context.RequestContext(overwrite=False)
    self.assertIs(context.get_current(), ctx1)