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
def test_find_policy_file_with_multiple_config_dirs(self):
    dir1 = tempfile.mkdtemp()
    self.tempdirs.append(dir1)
    dir2 = tempfile.mkdtemp()
    self.tempdirs.append(dir2)
    self.conf(['--config-dir', dir1, '--config-dir', dir2])
    self.assertEqual(2, len(self.conf.config_dirs))
    self.assertEqual(dir1, self.conf.config_dirs[0])
    self.assertEqual(dir2, self.conf.config_dirs[1])