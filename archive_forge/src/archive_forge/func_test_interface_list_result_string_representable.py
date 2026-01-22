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
def test_interface_list_result_string_representable(self):
    """Test for bugs.launchpad.net/python-novaclient/+bug/1280453."""
    interface_list = [{'net_id': 'd7745cf5-63f9-4883-b0ae-983f061e4f23', 'port_id': 'f35079da-36d5-4513-8ec1-0298d703f70e', 'mac_addr': 'fa:16:3e:4c:37:c8', 'port_state': 'ACTIVE', 'fixed_ips': [{'subnet_id': 'f1ad93ad-2967-46ba-b403-e8cbbe65f7fa', 'ip_address': '10.2.0.96'}]}]
    s = servers.Server(servers.ServerManager, interface_list[0], loaded=True)
    self.assertEqual('<Server: unknown-name>', '%r' % s)