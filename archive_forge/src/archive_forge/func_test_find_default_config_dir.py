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
def test_find_default_config_dir(self):
    paths = self.create_tempfiles([('def.conf.d/def', '[DEFAULT]')])
    p = os.path.dirname(paths[0])
    self.useFixture(fixtures.MonkeyPatch('oslo_config.cfg.find_config_dirs', lambda project, prog: p))
    self.conf(args=[], default_config_dirs=None)
    self.assertEqual([p], self.conf.config_dir)