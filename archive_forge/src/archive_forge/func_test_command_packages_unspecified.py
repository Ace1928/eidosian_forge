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
def test_command_packages_unspecified(self):
    sys.argv.append('build')
    d = self.create_distribution()
    self.assertEqual(d.get_command_packages(), ['distutils.command'])