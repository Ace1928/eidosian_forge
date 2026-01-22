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
def test_show_help(self):
    self.addCleanup(log.set_threshold, log._global_log.threshold)
    dist = Distribution()
    sys.argv = []
    dist.help = 1
    dist.script_name = 'setup.py'
    with captured_stdout() as s:
        dist.parse_command_line()
    output = [line for line in s.getvalue().split('\n') if line.strip() != '']
    self.assertTrue(output)