import json
import copy
import tempfile
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.glacier.layer1 import Layer1
from boto.compat import six
def test_get_archive_output(self):
    header = [('Content-Type', 'application/octet-stream')]
    self.set_http_response(status_code=200, header=header, body=self.job_content)
    response = self.service_connection.get_job_output(self.vault_name, 'example-job-id')
    self.assertEqual(self.job_content, response.read())