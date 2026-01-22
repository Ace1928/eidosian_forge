import io
import uuid
from keystoneauth1 import fixture
from keystoneauth1 import plugin as ksa_plugin
from keystoneauth1 import session
from oslo_log import log as logging
from requests_mock.contrib import fixture as rm_fixture
from keystonemiddleware.auth_token import _auth
from keystonemiddleware.tests.unit import utils
def test_auth_uri_from_fragments(self):
    auth_protocol = 'http'
    auth_host = 'testhost'
    auth_port = 8888
    auth_admin_prefix = 'admin'
    expected = '%s://%s:%d/admin' % (auth_protocol, auth_host, auth_port)
    plugin = self.new_plugin(auth_host=auth_host, auth_protocol=auth_protocol, auth_port=auth_port, auth_admin_prefix=auth_admin_prefix)
    endpoint = plugin.get_endpoint(self.session, interface=ksa_plugin.AUTH_INTERFACE)
    self.assertEqual(expected, endpoint)