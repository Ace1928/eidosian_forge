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
def test_find_default_config_file(self):
    paths = self.create_tempfiles([('def', '[DEFAULT]')])
    self.useFixture(fixtures.MonkeyPatch('oslo_config.cfg.find_config_files', lambda project, prog: paths))
    self.conf(args=[], default_config_files=None)
    self.assertEqual(paths, self.conf.config_file)