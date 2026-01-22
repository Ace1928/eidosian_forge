from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import data_sources as api_ds
from saharaclient.osc.v1 import data_sources as osc_ds
from saharaclient.tests.unit.osc.v1 import test_data_sources as tds_v1
def test_data_sources_update_required_options(self):
    arglist = ['source']
    verifylist = [('data_source', 'source')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.ds_mock.update.assert_called_once_with('id', {})
    expected_columns = ('Description', 'Id', 'Is protected', 'Is public', 'Name', 'Type', 'Url')
    self.assertEqual(expected_columns, columns)
    expected_data = ('Data Source for tests', 'id', True, True, 'source', 'swift', 'swift://container.sahara/object')
    self.assertEqual(expected_data, data)