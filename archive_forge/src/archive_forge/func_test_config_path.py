import os
import sys
from .. import bedding, osutils, tests
def test_config_path(self):
    self.assertIsSameRealPath(bedding.config_path(), self.appdata_bzr + '/bazaar.conf')
    self.overrideAttr(win32utils, 'get_appdata_location', lambda: None)
    self.assertRaises(RuntimeError, bedding.config_path)