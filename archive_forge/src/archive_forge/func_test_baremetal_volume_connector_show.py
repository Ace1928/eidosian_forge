import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_connector as bm_vol_connector
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_volume_connector_show(self):
    arglist = ['vvv-cccccc-vvvv']
    verifylist = [('volume_connector', baremetal_fakes.baremetal_volume_connector_uuid)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    args = ['vvv-cccccc-vvvv']
    self.baremetal_mock.volume_connector.get.assert_called_once_with(*args, fields=None)
    collist = ('connector_id', 'extra', 'node_uuid', 'type', 'uuid')
    self.assertEqual(collist, columns)
    datalist = (baremetal_fakes.baremetal_volume_connector_connector_id, baremetal_fakes.baremetal_volume_connector_extra, baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_volume_connector_type, baremetal_fakes.baremetal_volume_connector_uuid)
    self.assertEqual(datalist, tuple(data))