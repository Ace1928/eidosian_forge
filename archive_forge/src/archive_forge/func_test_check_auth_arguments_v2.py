import io
from requests_mock.contrib import fixture
import testtools
from barbicanclient import barbican as barb
from barbicanclient.barbican import Barbican
from barbicanclient import client
from barbicanclient import exceptions
from barbicanclient.tests import keystone_client_fixtures
def test_check_auth_arguments_v2(self):
    args = "--os-username 'bob' --os-password 'jan' --os-auth-url 'boop' --os-tenant-id 123 --os-identity-api-version '2.0'"
    argv, remainder = self.parser.parse_known_args(args.split())
    api_version = argv.os_identity_api_version
    barbican = Barbican()
    response = barbican.check_auth_arguments(argv, api_version)
    self.assertTrue(response)