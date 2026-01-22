import json
import copy
import tempfile
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.glacier.layer1 import Layer1
from boto.compat import six
def test_create_vault_parameters(self):
    self.set_http_response(status_code=201)
    self.service_connection.create_vault(self.vault_name)