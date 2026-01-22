import http.client as http
from oslo_utils import encodeutils
from glance.common import exception
from glance.tests import utils as test_utils
def test_non_unicode_error_msg(self):
    exc = exception.GlanceException('test')
    self.assertIsInstance(encodeutils.exception_to_unicode(exc), str)