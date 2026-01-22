from unittest import mock
import uuid
from oslo_config import cfg
from oslo_config import fixture as config
import stevedore
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.loading._plugins.identity import v2
from keystoneauth1.loading._plugins.identity import v3
from keystoneauth1.tests.unit.loading import utils
def test_get_named(self):
    loaded_opts = loading.get_plugin_options('v2password')
    plugin_opts = v2.Password().get_options()
    loaded_names = set([o.name for o in loaded_opts])
    plugin_names = set([o.name for o in plugin_opts])
    self.assertEqual(plugin_names, loaded_names)