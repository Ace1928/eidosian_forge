import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_target as bm_vol_target
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_volume_target_unset_property(self):
    arglist = [baremetal_fakes.baremetal_volume_target_uuid, '--property', 'key11']
    verifylist = [('volume_target', baremetal_fakes.baremetal_volume_target_uuid), ('properties', ['key11'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.volume_target.update.assert_called_once_with(baremetal_fakes.baremetal_volume_target_uuid, [{'path': '/properties/key11', 'op': 'remove'}])