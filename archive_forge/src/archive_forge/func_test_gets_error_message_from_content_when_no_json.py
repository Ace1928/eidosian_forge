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
def test_gets_error_message_from_content_when_no_json(self):
    resp = mock.MagicMock()
    resp.json.side_effect = ValueError()
    resp.content = content = 'content'
    msg = self.httpclient._get_error_message(resp)
    self.assertEqual(content, msg)