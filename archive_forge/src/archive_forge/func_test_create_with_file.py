import os.path
import pkg_resources as pkg
from urllib import parse
from urllib import request
from mistralclient.api.v2 import workflows
from mistralclient.tests.unit.v2 import base
def test_create_with_file(self):
    self.requests_mock.post(self.TEST_URL + URL_TEMPLATE_SCOPE, json={'workflows': [WORKFLOW]}, status_code=201)
    path = pkg.resource_filename('mistralclient', 'tests/unit/resources/wf_v2.yaml')
    wfs = self.workflows.create(path)
    self.assertIsNotNone(wfs)
    self.assertEqual(WF_DEF, wfs[0].definition)
    last_request = self.requests_mock.last_request
    self.assertEqual(WF_DEF, last_request.text)
    self.assertEqual('text/plain', last_request.headers['content-type'])