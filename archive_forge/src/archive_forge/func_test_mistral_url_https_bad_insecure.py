import os
import tempfile
from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from oslotest import base
import osprofiler.profiler
from mistralclient.api import client
@mock.patch('logging.Logger.warning')
@mock.patch('keystoneauth1.session.Session')
def test_mistral_url_https_bad_insecure(self, session_mock, log_warning_mock):
    fd, path = tempfile.mkstemp(suffix='.pem')
    keystone_client_instance = self.setup_keystone_mock(session_mock)
    try:
        client.client(mistral_url=MISTRAL_HTTPS_URL, user_id=keystone_client_instance.user_id, project_id=keystone_client_instance.project_id, api_key='password', user_domain_name='Default', project_domain_name='Default', auth_url=AUTH_HTTP_URL_v3, cacert=path, insecure=True)
    finally:
        os.close(fd)
        os.unlink(path)
    self.assertTrue(log_warning_mock.called)