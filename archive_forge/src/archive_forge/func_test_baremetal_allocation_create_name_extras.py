import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_allocation
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_allocation_create_name_extras(self):
    arglist = ['--resource-class', baremetal_fakes.baremetal_resource_class, '--uuid', baremetal_fakes.baremetal_uuid, '--name', baremetal_fakes.baremetal_name, '--extra', 'key1=value1', '--extra', 'key2=value2']
    verifylist = [('resource_class', baremetal_fakes.baremetal_resource_class), ('uuid', baremetal_fakes.baremetal_uuid), ('name', baremetal_fakes.baremetal_name), ('extra', ['key1=value1', 'key2=value2'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    args = {'resource_class': baremetal_fakes.baremetal_resource_class, 'uuid': baremetal_fakes.baremetal_uuid, 'name': baremetal_fakes.baremetal_name, 'extra': {'key1': 'value1', 'key2': 'value2'}}
    self.baremetal_mock.allocation.create.assert_called_once_with(**args)