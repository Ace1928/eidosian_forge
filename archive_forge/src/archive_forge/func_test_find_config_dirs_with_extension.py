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
def test_find_config_dirs_with_extension(self):
    config_dirs = ['/etc/foo.json.d']
    self.useFixture(fixtures.MonkeyPatch('sys.argv', ['foo']))
    self.useFixture(fixtures.MonkeyPatch('os.path.exists', lambda p: p in config_dirs))
    self.assertEqual(cfg.find_config_dirs(project='blaa'), [])
    self.assertEqual(cfg.find_config_dirs(project='blaa', extension='.json.d'), config_dirs)