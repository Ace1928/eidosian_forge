import io
from requests_mock.contrib import fixture
import testtools
from barbicanclient import barbican as barb
from barbicanclient.barbican import Barbican
from barbicanclient import client
from barbicanclient import exceptions
from barbicanclient.tests import keystone_client_fixtures
def test_should_fail_missing_credentials(self):
    message = 'ERROR: please specify authentication credentials'
    args = ''
    argv, remainder = self.parser.parse_known_args(args.split())
    e = self.assertRaises(Exception, self.barbican.create_client, argv)
    self.assertIn(message, str(e))