import fnmatch
import os
import re
import sys
import pkg_resources
from . import copydir
from . import pluginlib
from .command import Command, BadCommand
def display_vars(self, vars):
    vars = sorted(vars.items())
    print('Variables:')
    max_var = max([len(n) for n, v in vars])
    for name, value in vars:
        print('  %s:%s  %s' % (name, ' ' * (max_var - len(name)), value))