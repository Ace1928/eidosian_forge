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
def test_keywords_invalid_type(self):
    attrs = {'name': 'Monty', 'version': '1.0', 'keywords': ('spam', 'eggs', 'life of brian')}
    with captured_stderr() as error:
        d = Distribution(attrs)
    self.assertIn('should be a list', error.getvalue())
    self.assertIsInstance(d.metadata.keywords, list)
    self.assertEqual(d.metadata.keywords, list(attrs['keywords']))