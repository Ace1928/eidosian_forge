from unittest import mock
from openstackclient.common import configuration
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
@mock.patch('keystoneauth1.loading.base.get_plugin_options', return_value=opts)
def test_show_mask_with_cloud_config(self, m_get_plugin_opts):
    arglist = ['--mask']
    verifylist = [('mask', True)]
    self.app.client_manager.configuration_type = 'cloud_config'
    cmd = configuration.ShowConfiguration(self.app, None)
    parsed_args = self.check_parser(cmd, arglist, verifylist)
    columns, data = cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, data)