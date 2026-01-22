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
def test_config_file_tilde(self):
    homedir = os.path.expanduser('~')
    tmpfile = tempfile.mktemp(dir=homedir, prefix='cfg-', suffix='.conf')
    tmpbase = os.path.basename(tmpfile)
    try:
        self.conf(['--config-file', os.path.join('~', tmpbase)])
    except cfg.ConfigFilesNotFoundError as cfnfe:
        self.assertIn(homedir, str(cfnfe))
    self.useFixture(fixtures.MonkeyPatch('os.path.exists', lambda p: p == tmpfile))
    self.assertEqual(tmpfile, self.conf.find_file(tmpbase))