from oslo_utils import encodeutils
from oslotest import base
import glance_store
def test_backend_exception(self):
    msg = glance_store.BackendException()
    self.assertIn('', encodeutils.exception_to_unicode(msg))