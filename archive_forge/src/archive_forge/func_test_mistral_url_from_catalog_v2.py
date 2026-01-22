import os
import tempfile
from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from oslotest import base
import osprofiler.profiler
from mistralclient.api import client
@mock.patch('keystoneauth1.session.Session')
def test_mistral_url_from_catalog_v2(self, session_mock):
    session = mock.Mock()
    session_mock.side_effect = [session]
    get_endpoint = mock.Mock(return_value='http://mistral_host:8989/v2')
    session.get_endpoint = get_endpoint
    mistralclient = client.client(username='mistral', project_name='mistral', api_key='password', auth_url=AUTH_HTTP_URL_v2_0, service_type='workflowv2')
    self.assertEqual('http://mistral_host:8989/v2', mistralclient.actions.http_client.base_url)