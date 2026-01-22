from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient.osc.v1 import baremetal_create
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
from ironicclient.v1 import create_resources
@mock.patch.object(create_resources, 'create_resources', autospec=True)
def test_baremetal_create_resource_files(self, mock_create):
    arglist = ['file.yaml', 'file.json']
    verifylist = [('resource_files', ['file.yaml', 'file.json'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    mock_create.assert_called_once_with(self.app.client_manager.baremetal, ['file.yaml', 'file.json'])