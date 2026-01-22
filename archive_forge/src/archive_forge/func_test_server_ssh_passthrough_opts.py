import collections
import copy
import getpass
import json
import tempfile
from unittest import mock
from unittest.mock import call
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
def test_server_ssh_passthrough_opts(self, mock_exec):
    arglist = [self.server.name, '--', '-l', 'username', '-p', '2222']
    verifylist = [('server', self.server.name), ('login', None), ('port', None), ('identity', None), ('option', None), ('ipv4', False), ('ipv6', False), ('address_type', 'public'), ('verbose', False), ('ssh_args', ['-l', 'username', '-p', '2222'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch.object(self.cmd.log, 'warning') as mock_warning:
        result = self.cmd.take_action(parsed_args)
    self.assertIsNone(result)
    mock_exec.assert_called_once_with('ssh 192.168.1.30 -l username -p 2222')
    mock_warning.assert_not_called()