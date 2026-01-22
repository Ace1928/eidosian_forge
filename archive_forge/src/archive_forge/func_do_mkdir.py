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
def do_mkdir(self, test_case, input, args):
    if not args or len(args) != 1:
        raise SyntaxError('Usage: mkdir dir')
    d = args[0]
    self._ensure_in_jail(test_case, d)
    os.mkdir(d)
    return (0, None, None)