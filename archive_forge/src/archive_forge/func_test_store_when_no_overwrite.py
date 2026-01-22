import fixtures
import hashlib
import uuid
import warnings
from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
def test_store_when_no_overwrite(self):
    ctx = context.RequestContext(overwrite=False)
    self.assertIs(context.get_current(), ctx)