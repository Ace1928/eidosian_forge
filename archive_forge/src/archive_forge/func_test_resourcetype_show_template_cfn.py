from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import resource_type
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import resource_types
def test_resourcetype_show_template_cfn(self):
    arglist = ['OS::Heat::None', '--template-type', 'cfn']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.cmd.take_action(parsed_args)
    self.mock_client.resource_types.generate_template.assert_called_with(**{'resource_type': 'OS::Heat::None', 'template_type': 'cfn'})