import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc import utils
from manilaclient.osc.v2 import services as osc_services
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.data('2.82', '2.83')
def test_share_service_list(self, version):
    self.app.client_manager.share.api_version = api_versions.APIVersion(version)
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.services_mock.list.assert_called_with(search_opts={'host': None, 'binary': None, 'status': None, 'state': None, 'zone': None})
    if api_versions.APIVersion(version) >= api_versions.APIVersion('2.83'):
        self.assertEqual(self.column_headers_with_reason, columns)
        self.assertEqual(list(self.values_with_reason), list(data))
    else:
        self.assertEqual(self.column_headers, columns)
        self.assertEqual(list(self.values), list(data))