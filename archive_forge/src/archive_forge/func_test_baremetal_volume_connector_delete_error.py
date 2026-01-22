import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_connector as bm_vol_connector
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_volume_connector_delete_error(self):
    arglist = [baremetal_fakes.baremetal_volume_connector_uuid]
    verifylist = [('volume_connectors', [baremetal_fakes.baremetal_volume_connector_uuid])]
    self.baremetal_mock.volume_connector.delete.side_effect = exc.NotFound()
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exc.ClientException, self.cmd.take_action, parsed_args)
    self.baremetal_mock.volume_connector.delete.assert_called_with(baremetal_fakes.baremetal_volume_connector_uuid)