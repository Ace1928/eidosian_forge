import base64
import io
import os
import tempfile
from unittest import mock
from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import floatingips
from novaclient.tests.unit.fixture_data import servers as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import servers
def test_get_rdp_console(self):
    s = self.cs.servers.get(1234)
    rc = s.get_rdp_console('rdp-html5')
    self.assert_request_id(rc, fakes.FAKE_REQUEST_ID_LIST)
    self.assert_called('POST', '/servers/1234/remote-consoles')
    rc = self.cs.servers.get_rdp_console(s, 'rdp-html5')
    self.assert_request_id(rc, fakes.FAKE_REQUEST_ID_LIST)
    self.assert_called('POST', '/servers/1234/remote-consoles')
    self.assertRaises(exceptions.UnsupportedConsoleType, s.get_rdp_console, 'invalid')