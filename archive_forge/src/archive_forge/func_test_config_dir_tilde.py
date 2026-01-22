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
def test_config_dir_tilde(self):
    homedir = os.path.expanduser('~')
    try:
        tmpdir = tempfile.mkdtemp(dir=homedir, prefix='cfg-', suffix='.d')
        tmpfile = os.path.join(tmpdir, 'foo.conf')
        self.useFixture(fixtures.MonkeyPatch('glob.glob', lambda p: [tmpfile]))
        e = self.assertRaises(cfg.ConfigFilesNotFoundError, self.conf, ['--config-dir', os.path.join('~', os.path.basename(tmpdir))])
        self.assertIn(tmpdir, str(e))
    finally:
        try:
            shutil.rmtree(tmpdir)
        except OSError as exc:
            if exc.errno != 2:
                raise