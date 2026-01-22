from unittest import mock
from keystoneauth1 import session
from requests_mock.contrib import fixture
import testtools
from barbicanclient import client
from barbicanclient import exceptions
from barbicanclient.exceptions import UnsupportedVersion
from barbicanclient.tests.utils import get_server_supported_versions
from barbicanclient.tests.utils import get_version_endpoint
from barbicanclient.tests.utils import mock_session
from barbicanclient.tests.utils import mock_session_get
from barbicanclient.tests.utils import mock_session_get_endpoint
def test_fails_requesting_higher_microversion_than_supported_by_server(self):
    requested_version = '1.1'
    server_max_version = (1, 0)
    server_min_version = (1, 0)
    sess = self._create_mock_session(requested_version, server_max_version, server_min_version, self.endpoint)
    self.assertRaises(UnsupportedVersion, client.Client, session=sess, endpoint=self.endpoint, microversion=requested_version)