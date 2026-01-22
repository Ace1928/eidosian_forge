from unittest import mock
from oslo_serialization import jsonutils as json
from saharaclient.api import plugins as api_plugins
from saharaclient.osc.v1 import plugins as osc_plugins
from saharaclient.tests.unit.osc.v1 import fakes
@mock.patch('oslo_serialization.jsonutils.dump')
def test_get_plugin_configs_default_file(self, p_dump):
    m_open = mock.mock_open()
    with mock.patch('builtins.open', m_open, create=True):
        arglist = ['fake', '0.1']
        verifylist = [('plugin', 'fake'), ('plugin_version', '0.1')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.plugins_mock.get_version_details.assert_called_once_with('fake', '0.1')
        args_to_dump = p_dump.call_args[0]
        self.assertEqual(PLUGIN_INFO, args_to_dump[0])
        self.assertEqual('fake-0.1', m_open.call_args[0][0])