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
def test_project_id_is_added_to_default_headers(self):
    c = client._HTTPClient(session=self.session, microversion=_DEFAULT_MICROVERSION, endpoint=self.endpoint, project_id=self.project_id)
    self.assertIn('X-Project-Id', c._default_headers.keys())
    self.assertEqual(self.project_id, c._default_headers['X-Project-Id'])