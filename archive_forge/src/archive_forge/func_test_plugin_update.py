from unittest import mock
from oslo_serialization import jsonutils as json
from saharaclient.api import plugins as api_plugins
from saharaclient.osc.v1 import plugins as osc_plugins
from saharaclient.tests.unit.osc.v1 import fakes
@mock.patch('osc_lib.utils.read_blob_file_contents')
def test_plugin_update(self, read):
    arglist = ['fake', 'update.json']
    verifylist = [('plugin', 'fake'), ('json', 'update.json')]
    value = {'plugin_labels': {'enabled': {'status': True}}}
    value = json.dumps(value)
    read.return_value = value
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.plugins_mock.update.assert_called_once_with('fake', {'plugin_labels': {'enabled': {'status': True}}})
    expected_columns = ('Description', 'Name', 'Title', 'Versions', '', 'Plugin version 0.1: enabled', 'Plugin: enabled')
    self.assertEqual(expected_columns, columns)
    expected_data = ('Plugin for tests', 'fake', 'Fake Plugin', '0.1, 0.2', '', True, True)
    self.assertEqual(expected_data, data)