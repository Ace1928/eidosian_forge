import copy
from unittest import mock
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import software_deployment
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import software_configs
from heatclient.v1 import software_deployments
def test_deployment_create_no_signal(self):
    arglist = ['my_deploy', '--server', self.server_id, '--signal-transport', 'NO_SIGNAL']
    config = copy.deepcopy(self.config_defaults)
    config['inputs'] = config['inputs'][:-2]
    config['inputs'][2]['value'] = 'NO_SIGNAL'
    parsed_args = self.check_parser(self.cmd, arglist, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.config_client.create.assert_called_with(**config)
    self.sd_client.create.assert_called_with(**self.deploy_defaults)