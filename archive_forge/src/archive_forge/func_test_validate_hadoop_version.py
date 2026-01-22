from unittest import mock
import uuid
from saharaclient.api import base as sahara_base
from heat.common import exception
from heat.engine.clients.os import sahara
from heat.tests import common
from heat.tests import utils
def test_validate_hadoop_version(self):
    """Tests the validate_hadoop_version function."""
    versions = ['1.2.1', '2.6.0', '2.7.1']
    plugin_name = 'vanilla'
    self.my_plugin.name = plugin_name
    self.my_plugin.versions = versions
    self.sahara_client.plugins.get.return_value = self.my_plugin
    self.assertIsNone(self.sahara_plugin.validate_hadoop_version(plugin_name, '2.6.0'))
    ex = self.assertRaises(exception.StackValidationFailed, self.sahara_plugin.validate_hadoop_version, plugin_name, '1.2.3')
    self.assertEqual("Requested plugin 'vanilla' doesn't support version '1.2.3'. Allowed versions are 1.2.1, 2.6.0, 2.7.1", str(ex))
    calls = [mock.call(plugin_name), mock.call(plugin_name)]
    self.sahara_client.plugins.get.assert_has_calls(calls)