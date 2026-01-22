import re
import sys
import breezy
from ... import cmdline, commands, config, help_topics, option, plugin
def command_names(self):
    return ' '.join(self.data.all_command_aliases())