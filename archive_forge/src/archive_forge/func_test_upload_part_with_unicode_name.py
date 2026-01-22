import json
import copy
import tempfile
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.glacier.layer1 import Layer1
from boto.compat import six
def test_upload_part_with_unicode_name(self):
    fake_data = b'\xe2'
    self.set_http_response(status_code=204)
    self.service_connection.upload_part(u'unicode_vault_name', 'upload_id', 'linear_hash', 'tree_hash', (1, 2), fake_data)
    self.assertEqual(self.actual_request.path, '/-/vaults/unicode_vault_name/multipart-uploads/upload_id')
    self.assertIsInstance(self.actual_request.body, six.binary_type)
    self.assertEqual(self.actual_request.body, fake_data)