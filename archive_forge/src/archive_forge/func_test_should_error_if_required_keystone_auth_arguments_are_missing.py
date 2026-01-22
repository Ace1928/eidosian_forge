import io
from requests_mock.contrib import fixture
import testtools
from barbicanclient import barbican as barb
from barbicanclient.barbican import Barbican
from barbicanclient import client
from barbicanclient import exceptions
from barbicanclient.tests import keystone_client_fixtures
def test_should_error_if_required_keystone_auth_arguments_are_missing(self):
    expected_error_msg = 'ERROR: please specify the following --os-project-id or (--os-project-name and --os-project-domain-name) or  (--os-project-name and --os-project-domain-id)'
    self.assert_client_raises('--os-auth-url http://localhost:35357/v2.0 secret list', expected_error_msg)
    self.assert_client_raises('--os-auth-url http://localhost:35357/v2.0 --os-username barbican --os-password barbican secret list', expected_error_msg)