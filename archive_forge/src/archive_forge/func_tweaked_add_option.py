import re
import sys
import breezy
from ... import cmdline, commands, config, help_topics, option, plugin
def tweaked_add_option(*opts, **attrs):
    for name in opts:
        optswitches[name] = OptionData(name)