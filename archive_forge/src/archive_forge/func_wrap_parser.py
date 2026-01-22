import re
import sys
import breezy
from ... import cmdline, commands, config, help_topics, option, plugin
def wrap_parser(self, optswitches, parser):
    orig_add_option_group = parser.add_option_group

    def tweaked_add_option_group(*opts, **attrs):
        return self.wrap_container(optswitches, orig_add_option_group(*opts, **attrs))
    parser.add_option_group = tweaked_add_option_group
    return self.wrap_container(optswitches, parser)