import os
import tempfile
from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from oslotest import base
import osprofiler.profiler
from mistralclient.api import client
@mock.patch('mistralclient.auth.get_auth_handler')
def test_mistral_no_auth(self, get_auth_handler_mock):
    client.client(username='mistral', project_name='mistral', api_key='password', service_type='workflowv2')
    self.assertEqual(0, get_auth_handler_mock.call_count)