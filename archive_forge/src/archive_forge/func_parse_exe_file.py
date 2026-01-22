import pkg_resources
import sys
import optparse
from . import bool_optparse
import os
import re
import textwrap
from . import pluginlib
import configparser
import getpass
from logging.config import fileConfig
def parse_exe_file(config):
    import shlex
    p = configparser.RawConfigParser()
    p.read([config])
    command_name = 'exe'
    options = []
    if p.has_option('exe', 'command'):
        command_name = p.get('exe', 'command')
    if p.has_option('exe', 'options'):
        options = shlex.split(p.get('exe', 'options'))
    if p.has_option('exe', 'sys.path'):
        paths = shlex.split(p.get('exe', 'sys.path'))
        paths = [os.path.abspath(os.path.join(os.path.dirname(config), p)) for p in paths]
        for path in paths:
            pkg_resources.working_set.add_entry(path)
            sys.path.insert(0, path)
    args = [command_name, config] + options
    return args