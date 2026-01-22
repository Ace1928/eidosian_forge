import copy
import io
import json
import sys
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient.common import utils as commonutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_node
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
from ironicclient.v1 import utils as v1_utils
def test_baremetal_show(self):
    arglist = ['xxx-xxxxxx-xxxx']
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    args = ['xxx-xxxxxx-xxxx']
    self.baremetal_mock.node.get.assert_called_with(*args, fields=None)
    collist = ('chassis_uuid', 'instance_uuid', 'maintenance', 'name', 'power_state', 'provision_state', 'uuid')
    self.assertEqual(collist, columns)
    self.assertNotIn('ports', columns)
    self.assertNotIn('states', columns)
    datalist = (baremetal_fakes.baremetal_chassis_uuid_empty, baremetal_fakes.baremetal_instance_uuid, baremetal_fakes.baremetal_maintenance, baremetal_fakes.baremetal_name, baremetal_fakes.baremetal_power_state, baremetal_fakes.baremetal_provision_state, baremetal_fakes.baremetal_uuid)
    self.assertEqual(datalist, tuple(data))