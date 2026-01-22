import doctest
import os
import sys
from io import StringIO
import breezy
from .. import bedding, crash, osutils, plugin, tests
from . import features
def setup_fake_plugins(self):
    fake = plugin.PlugIn('fake_plugin', plugin)
    fake.version_info = lambda: (1, 2, 3)
    fake_plugins = {'fake_plugin': fake}
    self.overrideAttr(breezy.get_global_state(), 'plugins', fake_plugins)