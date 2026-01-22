import argparse
import errno
import functools
import io
import logging
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import fixtures
from oslotest import base
import testscenarios
from oslo_config import cfg
from oslo_config import types
def test_print_custom_usage(self):
    conf = self.TestConfigOpts()
    self.tempdirs = []
    f = io.StringIO()
    conf([], usage='%(prog)s FOO BAR')
    conf.print_usage(file=f)
    self.assertIn('usage: test FOO BAR', f.getvalue())
    self.assertNotIn('somedesc', f.getvalue())
    self.assertNotIn('tepilog', f.getvalue())
    self.assertNotIn('optional:', f.getvalue())