import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_connector as bm_vol_connector
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_volume_connector_set_type_and_connector_id(self):
    new_type = 'wwnn'
    new_conn_id = '11:22:33:44:55:66:77:88'
    arglist = [baremetal_fakes.baremetal_volume_connector_uuid, '--type', new_type, '--connector-id', new_conn_id]
    verifylist = [('volume_connector', baremetal_fakes.baremetal_volume_connector_uuid), ('type', new_type), ('connector_id', new_conn_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.volume_connector.update.assert_called_once_with(baremetal_fakes.baremetal_volume_connector_uuid, [{'path': '/type', 'value': new_type, 'op': 'add'}, {'path': '/connector_id', 'value': new_conn_id, 'op': 'add'}])