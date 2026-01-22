from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.network.v2 import port
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_set_port_mixed_binding_profile(self):
    arglist = ['--binding-profile', 'foo=bar', '--binding-profile', '{"foo2": "bar2"}', self._port.name]
    verifylist = [('binding_profile', {'foo': 'bar', 'foo2': 'bar2'}), ('port', self._port.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'binding:profile': {'foo': 'bar', 'foo2': 'bar2'}}
    self.network_client.update_port.assert_called_once_with(self._port, **attrs)
    self.assertIsNone(result)