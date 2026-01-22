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
def test_find_config_files_overrides(self):
    """Ensure priority of directories is enforced.

        Ensure we will only ever return two files: $project.conf and
        $prog.conf.
        """
    config_files = [os.path.expanduser('~/.foo/foo.conf'), os.path.expanduser('~/foo.conf'), os.path.expanduser('~/bar.conf'), '/etc/foo/foo.conf', '/etc/foo/bar.conf', '/etc/foo.conf', '/etc/bar.conf']
    self.useFixture(fixtures.MonkeyPatch('os.path.exists', lambda p: p in config_files))
    expected = [os.path.expanduser('~/.foo/foo.conf'), os.path.expanduser('~/bar.conf')]
    actual = cfg.find_config_files(project='foo', prog='bar')
    self.assertEqual(expected, actual)