import re
import sys
import breezy
from ... import cmdline, commands, config, help_topics, option, plugin
def brz_version(self):
    brz_version = breezy.version_string
    if not self.data.plugins:
        brz_version += '.'
    else:
        brz_version += ' and the following plugins:'
        for name, plugin in sorted(self.data.plugins.items()):
            brz_version += '\n# %s' % plugin
    return brz_version