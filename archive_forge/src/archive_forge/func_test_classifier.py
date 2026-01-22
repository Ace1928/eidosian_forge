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
def test_classifier(self):
    attrs = {'name': 'Boa', 'version': '3.0', 'classifiers': ['Programming Language :: Python :: 3']}
    dist = Distribution(attrs)
    self.assertEqual(dist.get_classifiers(), ['Programming Language :: Python :: 3'])
    meta = self.format_metadata(dist)
    self.assertIn('Metadata-Version: 1.1', meta)