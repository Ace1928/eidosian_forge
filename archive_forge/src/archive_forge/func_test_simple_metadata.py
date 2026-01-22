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
def test_simple_metadata(self):
    attrs = {'name': 'package', 'version': '1.0'}
    dist = Distribution(attrs)
    meta = self.format_metadata(dist)
    self.assertIn('Metadata-Version: 1.0', meta)
    self.assertNotIn('provides:', meta.lower())
    self.assertNotIn('requires:', meta.lower())
    self.assertNotIn('obsoletes:', meta.lower())