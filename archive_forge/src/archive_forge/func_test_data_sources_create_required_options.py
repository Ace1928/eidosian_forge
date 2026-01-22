from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import data_sources as api_ds
from saharaclient.osc.v1 import data_sources as osc_ds
from saharaclient.tests.unit.osc.v1 import test_data_sources as tds_v1
def test_data_sources_create_required_options(self):
    arglist = ['source', '--type', 'swift', '--url', 'swift://container.sahara/object']
    verifylist = [('name', 'source'), ('type', 'swift'), ('url', 'swift://container.sahara/object')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    called_args = {'credential_pass': None, 'credential_user': None, 'data_source_type': 'swift', 'name': 'source', 'description': '', 'url': 'swift://container.sahara/object', 'is_public': False, 'is_protected': False, 's3_credentials': None}
    self.ds_mock.create.assert_called_once_with(**called_args)
    expected_columns = ('Description', 'Id', 'Is protected', 'Is public', 'Name', 'Type', 'Url')
    self.assertEqual(expected_columns, columns)
    expected_data = ('Data Source for tests', 'id', True, True, 'source', 'swift', 'swift://container.sahara/object')
    self.assertEqual(expected_data, data)