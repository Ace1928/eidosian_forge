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
def test_post_checks_status_code(self):
    self.httpclient._check_status_code = mock.MagicMock()
    self.httpclient.post(path='secrets', json={'test_data': 'test'})
    self.httpclient._check_status_code.assert_has_calls([])