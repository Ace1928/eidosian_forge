import os
import io
import sys
import unittest
import warnings
import textwrap
from unittest import mock
from distutils.dist import Distribution, fix_help_options
from distutils.cmd import Command
from test.support import (
from test.support.os_helper import TESTFN
from distutils.tests import support
from distutils import log
def test_find_config_files_disable(self):
    temp_home = self.mkdtemp()
    if os.name == 'posix':
        user_filename = os.path.join(temp_home, '.pydistutils.cfg')
    else:
        user_filename = os.path.join(temp_home, 'pydistutils.cfg')
    with open(user_filename, 'w') as f:
        f.write('[distutils]\n')

    def _expander(path):
        return temp_home
    old_expander = os.path.expanduser
    os.path.expanduser = _expander
    try:
        d = Distribution()
        all_files = d.find_config_files()
        d = Distribution(attrs={'script_args': ['--no-user-cfg']})
        files = d.find_config_files()
    finally:
        os.path.expanduser = old_expander
    self.assertEqual(len(all_files) - 1, len(files))