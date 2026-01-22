import sys
from .. import plugin as _mod_plugin
from .. import symbol_versioning, tests
from . import features
def test_unavailable_plugin(self):
    feature = features.PluginLoadedFeature('idontexist')
    self.assertEqual('idontexist plugin', str(feature))
    self.assertFalse(feature.available())
    self.assertIs(None, feature.plugin)