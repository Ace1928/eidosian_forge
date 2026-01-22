import copy
from unittest import mock
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import resource
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import resources as v1_resources
def test_resource_show(self):
    arglist = ['my_stack', 'my_resource']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.resource_client.get.assert_called_with('my_stack', 'my_resource', with_attr=None)
    for key in self.response:
        self.assertIn(key, columns)
        self.assertIn(self.response[key], data)