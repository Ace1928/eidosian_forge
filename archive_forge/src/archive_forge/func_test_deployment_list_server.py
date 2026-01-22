import copy
from unittest import mock
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import software_deployment
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import software_configs
from heatclient.v1 import software_deployments
def test_deployment_list_server(self):
    kwargs = {}
    kwargs['server_id'] = 'ec14c864-096e-4e27-bb8a-2c2b4dc6f3f5'
    arglist = ['--server', 'ec14c864-096e-4e27-bb8a-2c2b4dc6f3f5']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.sd_client.list.assert_called_with(**kwargs)
    self.assertEqual(self.columns, columns)