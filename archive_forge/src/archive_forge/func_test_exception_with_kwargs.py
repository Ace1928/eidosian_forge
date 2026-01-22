from oslo_utils import encodeutils
from oslotest import base
import glance_store
def test_exception_with_kwargs(self):
    msg = glance_store.NotFound('Message: %(foo)s', foo='bar')
    self.assertIn('Message: bar', encodeutils.exception_to_unicode(msg))