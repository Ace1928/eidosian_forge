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
def test_conf_files_reload_default(self):
    self.conf.register_cli_opt(cfg.StrOpt('foo1'))
    self.conf.register_cli_opt(cfg.StrOpt('foo2'))
    paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo1 = default1\n'), ('2', '[DEFAULT]\nfoo2 = default2\n')])
    paths_change = self.create_tempfiles([('1', '[DEFAULT]\nfoo1 = change_default1\n'), ('2', '[DEFAULT]\nfoo2 = change_default2\n')])
    self.conf(args=[], default_config_files=paths)
    self.assertTrue(hasattr(self.conf, 'foo1'))
    self.assertEqual('default1', self.conf.foo1)
    self.assertTrue(hasattr(self.conf, 'foo2'))
    self.assertEqual('default2', self.conf.foo2)
    shutil.copy(paths_change[0], paths[0])
    shutil.copy(paths_change[1], paths[1])
    self.conf.reload_config_files()
    self.assertTrue(hasattr(self.conf, 'foo1'))
    self.assertEqual('change_default1', self.conf.foo1)
    self.assertTrue(hasattr(self.conf, 'foo2'))
    self.assertEqual('change_default2', self.conf.foo2)