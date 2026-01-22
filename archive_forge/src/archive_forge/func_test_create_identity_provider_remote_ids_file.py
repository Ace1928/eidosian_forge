import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v3 import identity_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as test_utils
def test_create_identity_provider_remote_ids_file(self):
    arglist = ['--remote-id-file', '/tmp/file_name', identity_fakes.idp_id]
    verifylist = [('identity_provider_id', identity_fakes.idp_id), ('remote_id_file', '/tmp/file_name')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    mocker = mock.Mock()
    mocker.return_value = '\n'.join(identity_fakes.idp_remote_ids)
    with mock.patch('openstackclient.identity.v3.identity_provider.utils.read_blob_file_contents', mocker):
        columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'remote_ids': identity_fakes.idp_remote_ids, 'description': None, 'domain_id': None, 'enabled': True}
    self.identity_providers_mock.create.assert_called_with(id=identity_fakes.idp_id, **kwargs)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.datalist, data)