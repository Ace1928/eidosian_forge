import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_connector as bm_vol_connector
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_volume_connector_show_fields_multiple(self):
    arglist = ['vvv-cccccc-vvvv', '--fields', 'uuid', 'connector_id', '--fields', 'type']
    verifylist = [('fields', [['uuid', 'connector_id'], ['type']]), ('volume_connector', baremetal_fakes.baremetal_volume_connector_uuid)]
    fake_vc = copy.deepcopy(baremetal_fakes.VOLUME_CONNECTOR)
    fake_vc.pop('extra')
    fake_vc.pop('node_uuid')
    self.baremetal_mock.volume_connector.get.return_value = baremetal_fakes.FakeBaremetalResource(None, fake_vc, loaded=True)
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    args = ['vvv-cccccc-vvvv']
    fields = ['uuid', 'connector_id', 'type']
    self.baremetal_mock.volume_connector.get.assert_called_once_with(*args, fields=fields)
    collist = ('connector_id', 'type', 'uuid')
    self.assertEqual(collist, columns)
    datalist = (baremetal_fakes.baremetal_volume_connector_connector_id, baremetal_fakes.baremetal_volume_connector_type, baremetal_fakes.baremetal_volume_connector_uuid)
    self.assertEqual(datalist, tuple(data))