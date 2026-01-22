import copy
from unittest import mock
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import software_deployment
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import software_configs
from heatclient.v1 import software_deployments
@mock.patch('heatclient.common.deployment_utils.build_signal_id', return_value='signal_id')
def test_deployment_create_input_value(self, mock_build):
    arglist = ['my_deploy', '--server', self.server_id, '--input-value', 'foo=bar']
    config = copy.deepcopy(self.config_defaults)
    config['inputs'].insert(0, {'name': 'foo', 'type': 'String', 'value': 'bar'})
    parsed_args = self.check_parser(self.cmd, arglist, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.config_client.create.assert_called_with(**config)
    self.sd_client.create.assert_called_with(**self.deploy_defaults)