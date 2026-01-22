import copy
from unittest import mock
import testscenarios
from heatclient import exc
from heatclient.osc.v1 import event
from heatclient.tests.unit.osc.v1 import fakes
from heatclient.v1 import events
def test_event_list_logical_resource_id(self):
    arglist = ['my_stack', '--format', 'table']
    del self.event.data['resource_name']
    cols = copy.deepcopy(self.fields)
    cols.pop()
    cols[0] = 'logical_resource_id'
    parsed_args = self.check_parser(self.cmd, arglist, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.event_client.list.assert_called_with(**self.defaults)
    self.assertEqual(cols, columns)
    self.event.data['resource_name'] = 'resource1'