from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient.common import cliutils
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_instances as osc_share_instances
from manilaclient import api_versions
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_instance_set_status(self):
    new_status = 'available'
    arglist = [self.share_instance.id, '--status', new_status]
    verifylist = [('instance', self.share_instance.id), ('status', new_status)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.instances_mock.reset_state.assert_called_with(self.share_instance, new_status)
    self.assertIsNone(result)