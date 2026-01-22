import fixtures
import hashlib
import uuid
import warnings
from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
def test_generate_request_id_unique(self):
    id1 = context.generate_request_id()
    id2 = context.generate_request_id()
    self.assertNotEqual(id1, id2)