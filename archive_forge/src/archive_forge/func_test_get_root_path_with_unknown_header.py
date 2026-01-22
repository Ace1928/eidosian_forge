import http.client as http_client
import httplib2
from oslo_serialization import jsonutils
from glance.tests import functional
from glance.tests.unit import test_versions as tv
def test_get_root_path_with_unknown_header(self):
    """Assert GET / with Accept: unknown header
        Verify version choices returned. Verify message in API log about
        unknown accept header.
        """
    path = 'http://%s:%d/' % ('127.0.0.1', self.api_port)
    http = httplib2.Http()
    headers = {'Accept': 'unknown'}
    response, content_json = http.request(path, 'GET', headers=headers)
    self.assertEqual(http_client.MULTIPLE_CHOICES, response.status)
    content = jsonutils.loads(content_json.decode())
    self.assertEqual(self.versions, content)