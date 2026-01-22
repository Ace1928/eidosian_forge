import os
from breezy import ignores, osutils
from breezy.tests import TestCaseWithMemoryTransport
from breezy.tests.features import PluginLoadedFeature
def test_export_pot_plugin_unknown(self):
    out, err = self.run_bzr('export-pot --plugin=lalalala', retcode=3)
    self.assertContainsRe(err, 'ERROR: Plugin lalalala is not loaded')