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
def test_announce(self):
    dist = Distribution()
    args = ('ok',)
    kwargs = {'level': 'ok2'}
    self.assertRaises(ValueError, dist.announce, args, kwargs)