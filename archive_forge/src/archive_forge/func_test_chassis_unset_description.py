import copy
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_chassis
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_chassis_unset_description(self):
    uuid = baremetal_fakes.baremetal_chassis_uuid
    arglist = [uuid, '--description']
    verifylist = [('chassis', uuid), ('description', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.chassis.update.assert_called_once_with(uuid, [{'path': '/description', 'op': 'remove'}])