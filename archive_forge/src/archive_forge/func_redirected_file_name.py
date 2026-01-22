import doctest
import errno
import glob
import logging
import os
import shlex
import sys
import textwrap
from .. import osutils, tests, trace
from ..tests import ui_testing
def redirected_file_name(direction, name, args):
    if name == '':
        try:
            name = args.pop(0)
        except IndexError:
            name = ''
    return name