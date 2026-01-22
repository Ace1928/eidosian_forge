import os.path
import pkg_resources as pkg
from urllib import parse
from urllib import request
from mistralclient.api.v2 import workflows
from mistralclient.tests.unit.v2 import base
def test_update_with_file_uri(self):
    self.requests_mock.put(self.TEST_URL + URL_TEMPLATE_SCOPE, json={'workflows': [WORKFLOW]})
    path = pkg.resource_filename('mistralclient', 'tests/unit/resources/wf_v2.yaml')
    path = os.path.abspath(path)
    uri = parse.urljoin('file:', request.pathname2url(path))
    wfs = self.workflows.update(uri)
    self.assertIsNotNone(wfs)
    self.assertEqual(WF_DEF, wfs[0].definition)
    last_request = self.requests_mock.last_request
    self.assertEqual(WF_DEF, last_request.text)
    self.assertEqual('text/plain', last_request.headers['content-type'])