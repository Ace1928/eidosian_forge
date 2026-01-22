from unittest import mock
from stevedore import enabled
from neutron_lib.tests import _base as base
from neutron_lib.utils import runtime
@mock.patch.object(runtime, 'LOG')
@mock.patch.object(enabled, 'EnabledExtensionManager')
def test_init_reload_no_plugins(self, mock_mgr, mock_log):
    mock_mgr().names.return_value = []
    plugins = runtime.NamespacedPlugins('_test_ns_')
    mock_log.debug.assert_called_once()
    mock_mgr().map.assert_not_called()
    self.assertDictEqual({}, plugins._extensions)