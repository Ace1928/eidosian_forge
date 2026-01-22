import sys
import os
import re
from email import message_from_file
from distutils.errors import *
from distutils.fancy_getopt import FancyGetopt, translate_longopt
from distutils.util import check_environ, strtobool, rfc822_escape
from distutils import log
from distutils.debug import DEBUG
def dump_option_dicts(self, header=None, commands=None, indent=''):
    from pprint import pformat
    if commands is None:
        commands = sorted(self.command_options.keys())
    if header is not None:
        self.announce(indent + header)
        indent = indent + '  '
    if not commands:
        self.announce(indent + 'no commands known yet')
        return
    for cmd_name in commands:
        opt_dict = self.command_options.get(cmd_name)
        if opt_dict is None:
            self.announce(indent + "no option dict for '%s' command" % cmd_name)
        else:
            self.announce(indent + "option dict for '%s' command:" % cmd_name)
            out = pformat(opt_dict)
            for line in out.split('\n'):
                self.announce(indent + '  ' + line)