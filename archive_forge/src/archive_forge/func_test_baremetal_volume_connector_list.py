import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_connector as bm_vol_connector
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_volume_connector_list(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'marker': None, 'limit': None}
    self.baremetal_mock.volume_connector.list.assert_called_once_with(**kwargs)
    collist = ('UUID', 'Node UUID', 'Type', 'Connector ID')
    self.assertEqual(collist, columns)
    datalist = ((baremetal_fakes.baremetal_volume_connector_uuid, baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_volume_connector_type, baremetal_fakes.baremetal_volume_connector_connector_id),)
    self.assertEqual(datalist, tuple(data))