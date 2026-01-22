import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_target as bm_vol_target
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_volume_target_create(self):
    arglist = ['--node', baremetal_fakes.baremetal_uuid, '--type', baremetal_fakes.baremetal_volume_target_volume_type, '--boot-index', baremetal_fakes.baremetal_volume_target_boot_index, '--volume-id', baremetal_fakes.baremetal_volume_target_volume_id, '--uuid', baremetal_fakes.baremetal_volume_target_uuid]
    verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('volume_type', baremetal_fakes.baremetal_volume_target_volume_type), ('boot_index', baremetal_fakes.baremetal_volume_target_boot_index), ('volume_id', baremetal_fakes.baremetal_volume_target_volume_id), ('uuid', baremetal_fakes.baremetal_volume_target_uuid)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    args = {'node_uuid': baremetal_fakes.baremetal_uuid, 'volume_type': baremetal_fakes.baremetal_volume_target_volume_type, 'boot_index': baremetal_fakes.baremetal_volume_target_boot_index, 'volume_id': baremetal_fakes.baremetal_volume_target_volume_id, 'uuid': baremetal_fakes.baremetal_volume_target_uuid}
    self.baremetal_mock.volume_target.create.assert_called_once_with(**args)