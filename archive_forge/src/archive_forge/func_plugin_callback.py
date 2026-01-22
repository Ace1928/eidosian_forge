import re
import sys
import breezy
from ... import cmdline, commands, config, help_topics, option, plugin
def plugin_callback(option, opt, value, parser):
    values = parser.values.selected_plugins
    if value == '-':
        del values[:]
    else:
        values.append(value)