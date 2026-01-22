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
@mock.patch.object(cliutils, 'print_list', mock.Mock())
def test_security_service_list_all_filters(self):
    filters = {'status': 'new', 'name': 'fake-name', 'type': 'ldap', 'user': 'fake-user', 'dns-ip': '1.1.1.1', 'ou': 'fake-ou', 'server': 'fake-server', 'domain': 'fake-domain', 'offset': 10, 'limit': 20}
    command_str = 'security-service-list'
    for key, value in filters.items():
        command_str += ' --%(key)s=%(value)s' % {'key': key, 'value': value}
    self.run_command(command_str)
    self.assert_called('GET', '/security-services?dns_ip=1.1.1.1&domain=fake-domain&limit=20&name=fake-name&offset=10&ou=fake-ou&server=fake-server&status=new&type=ldap&user=fake-user')
    cliutils.print_list.assert_called_once_with(mock.ANY, fields=['id', 'name', 'status', 'type'])