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
def test_str_sub_with_backslash_escape_char(self):
    self._prep_test_str_sub()
    paths = self.create_tempfiles([('test', '[DEFAULT]\nbar=foo-somethin\\$k2\n')])
    self.conf(['--config-file', paths[0]])
    self.assertTrue(hasattr(self.conf, 'bar'))
    self.assertEqual('foo-somethin$k2', self.conf.bar)