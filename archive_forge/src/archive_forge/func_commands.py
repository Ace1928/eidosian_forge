import re
import sys
import breezy
from ... import cmdline, commands, config, help_topics, option, plugin
def commands(self):
    for name in sorted(commands.all_command_names()):
        self.command(name)