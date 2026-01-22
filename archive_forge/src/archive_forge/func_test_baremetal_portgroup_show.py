import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_portgroup_show(self):
    arglist = ['ppp-gggggg-pppp']
    verifylist = [('portgroup', baremetal_fakes.baremetal_portgroup_uuid)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    args = ['ppp-gggggg-pppp']
    self.baremetal_mock.portgroup.get.assert_called_with(*args, fields=None)
    collist = ('address', 'extra', 'mode', 'name', 'node_uuid', 'properties', 'uuid')
    self.assertEqual(collist, columns)
    datalist = (baremetal_fakes.baremetal_portgroup_address, baremetal_fakes.baremetal_portgroup_extra, baremetal_fakes.baremetal_portgroup_mode, baremetal_fakes.baremetal_portgroup_name, baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_portgroup_properties, baremetal_fakes.baremetal_portgroup_uuid)
    self.assertEqual(datalist, tuple(data))