import os
import sys
from .. import bedding, osutils, tests
def test_authentication_config_path(self):
    self.assertIsSameRealPath(bedding.authentication_config_path(), self.appdata_bzr + '/authentication.conf')
    self.overrideAttr(win32utils, 'get_appdata_location', lambda: None)
    self.assertRaises(RuntimeError, bedding.authentication_config_path)