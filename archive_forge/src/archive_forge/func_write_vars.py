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
def write_vars(self, config, vars, section='pastescript'):
    """
        Given a configuration filename, this will add items in the
        vars mapping to the configuration file.  Will create the
        configuration file if it doesn't exist.
        """
    modified = False
    p = configparser.RawConfigParser()
    if not os.path.exists(config):
        f = open(config, 'w')
        f.write('')
        f.close()
        modified = True
    p.read([config])
    if not p.has_section(section):
        p.add_section(section)
        modified = True
    existing_options = p.options(section)
    for key, value in vars.items():
        if key not in existing_options and '%s__eval__' % key not in existing_options:
            if not isinstance(value, str):
                p.set(section, '%s__eval__' % key, repr(value))
            else:
                p.set(section, key, value)
            modified = True
    if modified:
        p.write(open(config, 'w'))