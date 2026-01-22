import os
import sys
from .. import bedding, osutils, tests
def test_ensure_config_dir_exists(self):
    xdgconfigdir = osutils.pathjoin(self.test_home_dir, 'xdgconfig')
    self.overrideEnv('XDG_CONFIG_HOME', xdgconfigdir)
    bedding.ensure_config_dir_exists()
    newdir = osutils.pathjoin(xdgconfigdir, 'breezy')
    self.assertTrue(os.path.isdir(newdir))