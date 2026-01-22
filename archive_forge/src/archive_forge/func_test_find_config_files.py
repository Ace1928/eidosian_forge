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
def test_find_config_files(self):
    config_files = [os.path.expanduser('~/.blaa/blaa.conf'), '/etc/foo.conf']
    self.useFixture(fixtures.MonkeyPatch('sys.argv', ['foo']))
    self.useFixture(fixtures.MonkeyPatch('os.path.exists', lambda p: p in config_files))
    self.assertEqual(cfg.find_config_files(project='blaa'), config_files)