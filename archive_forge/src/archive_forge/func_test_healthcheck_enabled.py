import http.client
import tempfile
import httplib2
from glance.tests import functional
from glance.tests import utils
@utils.skip_if_disabled
def test_healthcheck_enabled(self):
    self.cleanup()
    self.start_servers(**self.__dict__.copy())
    response, content = self.request()
    self.assertEqual(b'OK', content)
    self.assertEqual(http.client.OK, response.status)
    self.stop_servers()