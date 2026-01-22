import itertools
from unittest import mock
import ddt
import fixtures
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient import client
from manilaclient.common.apiclient import utils as apiclient_utils
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
from manilaclient import shell
from manilaclient.tests.unit import utils as test_utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient import utils
from manilaclient.v2 import messages
from manilaclient.v2 import security_services
from manilaclient.v2 import share_access_rules
from manilaclient.v2 import share_group_types
from manilaclient.v2 import share_groups
from manilaclient.v2 import share_instances
from manilaclient.v2 import share_network_subnets
from manilaclient.v2 import share_networks
from manilaclient.v2 import share_servers
from manilaclient.v2 import share_snapshots
from manilaclient.v2 import share_types
from manilaclient.v2 import shares
from manilaclient.v2 import shell as shell_v2
@ddt.data({'--name': 'fake_name'}, {'--description': 'fake_description'}, {'--dns-ip': 'fake_dns_ip'}, {'--ou': 'fake_ou'}, {'--domain': 'fake_domain'}, {'--server': 'fake_server'}, {'--user': 'fake_user'}, {'--password': 'fake_password'}, {'--name': 'fake_name', '--description': 'fake_description', '--dns-ip': 'fake_dns_ip', '--ou': 'fake_ou', '--domain': 'fake_domain', '--server': 'fake_server', '--user': 'fake_user', '--password': 'fake_password'}, {'--name': '""'}, {'--description': '""'}, {'--dns-ip': '""'}, {'--ou': '""'}, {'--domain': '""'}, {'--server': '""'}, {'--user': '""'}, {'--password': '""'}, {'--name': '""', '--description': '""', '--dns-ip': '""', '--ou': '""', '--domain': '""', '--server': '""', '--user': '""', '--password': '""'})
def test_security_service_update(self, data):
    cmd = 'security-service-update 1111'
    expected = dict()
    for k, v in data.items():
        cmd += ' ' + k + ' ' + v
        expected[k[2:].replace('-', '_')] = v
    expected = dict(security_service=expected)
    self.run_command(cmd)
    self.assert_called('PUT', '/security-services/1111', body=expected)