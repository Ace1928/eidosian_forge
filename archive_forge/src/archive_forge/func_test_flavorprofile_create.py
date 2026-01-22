import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import flavorprofile
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('octaviaclient.osc.v2.utils.get_flavorprofile_attrs')
def test_flavorprofile_create(self, mock_client):
    mock_client.return_value = self.flavorprofile_info
    arglist = ['--name', self._flavorprofile.name, '--provider', 'mock_provider', '--flavor-data', '{"mock_key": "mock_value"}']
    verifylist = [('provider', 'mock_provider'), ('name', self._flavorprofile.name), ('flavor_data', '{"mock_key": "mock_value"}')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.flavorprofile_create.assert_called_with(json={'flavorprofile': self.flavorprofile_info})