import os.path
import pkg_resources as pkg
from urllib import parse
from urllib import request
from mistralclient.api import base as api_base
from mistralclient.api.v2 import workbooks
from mistralclient.tests.unit.v2 import base
def test_validate_failed(self):
    mock_result = {'valid': False, 'error': "Task properties 'action' and 'workflow' can't be specified both"}
    self.requests_mock.post(self.TEST_URL + URL_TEMPLATE_VALIDATE, json=mock_result)
    result = self.workbooks.validate(INVALID_WB_DEF)
    self.assertIsNotNone(result)
    self.assertIn('valid', result)
    self.assertFalse(result['valid'])
    self.assertIn('error', result)
    self.assertIn("Task properties 'action' and 'workflow' can't be specified both", result['error'])
    last_request = self.requests_mock.last_request
    self.assertEqual(INVALID_WB_DEF, last_request.text)
    self.assertEqual('text/plain', last_request.headers['content-type'])