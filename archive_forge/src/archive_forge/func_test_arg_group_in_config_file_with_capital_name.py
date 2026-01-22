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
def test_arg_group_in_config_file_with_capital_name(self):
    self.conf.register_group(cfg.OptGroup('blaa'))
    self.conf.register_opt(cfg.StrOpt('foo'), group='blaa')
    paths = self.create_tempfiles([('test', '[BLAA]\nfoo = bar\n')])
    self.conf(['--config-file', paths[0]])
    self.assertFalse(hasattr(self.conf, 'BLAA'))
    self.assertTrue(hasattr(self.conf, 'blaa'))
    self.assertTrue(hasattr(self.conf.blaa, 'foo'))
    self.assertEqual('bar', self.conf.blaa.foo)