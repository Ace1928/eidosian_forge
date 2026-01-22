import fixtures
import hashlib
import uuid
import warnings
from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
def test_admin_context_show_deleted_flag_set(self):
    ctx = context.get_admin_context(show_deleted=True)
    self.assertTrue(ctx.is_admin)
    self.assertTrue(ctx.show_deleted)