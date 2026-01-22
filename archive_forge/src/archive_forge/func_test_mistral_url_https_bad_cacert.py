import os
import tempfile
from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from oslotest import base
import osprofiler.profiler
from mistralclient.api import client
@mock.patch('keystoneauth1.session.Session')
def test_mistral_url_https_bad_cacert(self, session_mock):
    keystone_client_instance = self.setup_keystone_mock(session_mock)
    self.assertRaises(ValueError, client.client, mistral_url=MISTRAL_HTTPS_URL, username='mistral', project_name='mistral', api_key='password', user_domain_name='Default', project_domain_name='Default', auth_url=AUTH_HTTP_URL_v3, cacert='/path/to/foobar', insecure=False)