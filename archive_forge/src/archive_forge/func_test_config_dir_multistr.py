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
def test_config_dir_multistr(self):
    self.conf.register_cli_opt(cfg.MultiStrOpt('foo'))
    dir = tempfile.mkdtemp()
    self.tempdirs.append(dir)
    paths = self.create_tempfiles([(os.path.join(dir, '00-test'), '[DEFAULT]\nfoo = bar-00\n'), (os.path.join(dir, '02-test'), '[DEFAULT]\nfoo = bar-02\n'), (os.path.join(dir, '01-test'), '[DEFAULT]\nfoo = bar-01\n')])
    self.conf(['--foo', 'bar', '--config-dir', os.path.dirname(paths[0])])
    self.assertTrue(hasattr(self.conf, 'foo'))
    self.assertEqual(['bar', 'bar-00', 'bar-01', 'bar-02'], self.conf.foo)