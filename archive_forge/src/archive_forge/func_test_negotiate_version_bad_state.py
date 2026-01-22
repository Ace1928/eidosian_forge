from http import client as http_client
import json
import time
from unittest import mock
from keystoneauth1 import exceptions as kexc
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient import exc
from ironicclient.tests.unit import utils
@mock.patch.object(filecache, 'save_data', autospec=True)
def test_negotiate_version_bad_state(self, mock_save_data):
    self.test_object.api_version_select_state = 'word of the day: augur'
    self.assertRaises(RuntimeError, self.test_object.negotiate_version, None, None)
    self.assertEqual(0, mock_save_data.call_count)