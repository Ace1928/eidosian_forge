import copy
from unittest import mock
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import resource
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import resources as v1_resources
@mock.patch('urllib.request.urlopen')
def test_resource_signal_file(self, urlopen):
    data = mock.Mock()
    data.read.side_effect = ['{"message":"Content"}']
    urlopen.return_value = data
    arglist = ['my_stack', 'my_resource', '--data-file', 'test_file']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.cmd.take_action(parsed_args)
    self.resource_client.signal.assert_called_with(**{'data': {'message': 'Content'}, 'stack_id': 'my_stack', 'resource_name': 'my_resource'})