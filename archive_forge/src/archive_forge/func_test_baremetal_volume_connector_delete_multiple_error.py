import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_connector as bm_vol_connector
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_volume_connector_delete_multiple_error(self):
    fake_volume_connector_uuid2 = 'vvv-cccccc-cccc'
    arglist = [baremetal_fakes.baremetal_volume_connector_uuid, fake_volume_connector_uuid2]
    verifylist = [('volume_connectors', [baremetal_fakes.baremetal_volume_connector_uuid, fake_volume_connector_uuid2])]
    self.baremetal_mock.volume_connector.delete.side_effect = [None, exc.NotFound()]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exc.ClientException, self.cmd.take_action, parsed_args)
    self.baremetal_mock.volume_connector.delete.assert_has_calls([mock.call(baremetal_fakes.baremetal_volume_connector_uuid), mock.call(fake_volume_connector_uuid2)])
    self.assertEqual(2, self.baremetal_mock.volume_connector.delete.call_count)