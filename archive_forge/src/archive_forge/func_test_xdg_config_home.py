import os
import sys
from .. import bedding, osutils, tests
def test_xdg_config_home(self):
    """When XDG_CONFIG_HOME is set, use it."""
    xdgconfigdir = osutils.pathjoin(self.test_home_dir, 'xdgconfig')
    self.overrideEnv('XDG_CONFIG_HOME', xdgconfigdir)
    newdir = osutils.pathjoin(xdgconfigdir, 'bazaar')
    os.makedirs(newdir)
    self.assertEqual(bedding.config_dir(), newdir)