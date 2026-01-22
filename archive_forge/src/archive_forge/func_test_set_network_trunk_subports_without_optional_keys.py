import copy
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
import testtools
from openstackclient.network.v2 import network_trunk
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_set_network_trunk_subports_without_optional_keys(self):
    subport = copy.copy(self._trunk['sub_ports'][0])
    subport.pop('segmentation_type')
    subport.pop('segmentation_id')
    arglist = ['--subport', 'port=%(port)s' % {'port': subport['port_id']}, self._trunk['name']]
    verifylist = [('trunk', self._trunk['name']), ('set_subports', [{'port': subport['port_id']}])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.network_client.add_trunk_subports.assert_called_once_with(self._trunk, [subport])
    self.assertIsNone(result)