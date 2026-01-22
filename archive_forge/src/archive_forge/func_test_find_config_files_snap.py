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
def test_find_config_files_snap(self):
    config_files = ['/snap/nova/current/etc/blaa/blaa.conf', '/var/snap/nova/common/etc/blaa/blaa.conf']
    fake_env = {'SNAP': '/snap/nova/current/', 'SNAP_COMMON': '/var/snap/nova/common/'}
    self.useFixture(fixtures.MonkeyPatch('sys.argv', ['foo']))
    self.useFixture(fixtures.MonkeyPatch('os.path.exists', lambda p: p in config_files))
    self.useFixture(fixtures.MonkeyPatch('os.environ', fake_env))
    self.assertEqual(cfg.find_config_files(project='blaa'), ['/var/snap/nova/common/etc/blaa/blaa.conf'])