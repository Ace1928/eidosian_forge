import os.path
import pkg_resources as pkg
from urllib import parse
from urllib import request
from mistralclient.api import base as api_base
from mistralclient.api.v2 import workbooks
from mistralclient.tests.unit.v2 import base
def test_create_with_file_uri(self):
    self.requests_mock.post(self.TEST_URL + URL_TEMPLATE, json=WORKBOOK, status_code=201)
    path = pkg.resource_filename('mistralclient', 'tests/unit/resources/wb_v2.yaml')
    path = os.path.abspath(path)
    uri = parse.urljoin('file:', request.pathname2url(path))
    wb = self.workbooks.create(uri)
    self.assertIsNotNone(wb)
    self.assertEqual(WB_DEF, wb.definition)
    last_request = self.requests_mock.last_request
    self.assertEqual(WB_DEF, last_request.text)
    self.assertEqual('text/plain', last_request.headers['content-type'])