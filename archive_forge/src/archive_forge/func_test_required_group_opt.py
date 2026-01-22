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
def test_required_group_opt(self):
    self.conf.register_group(cfg.OptGroup('blaa'))
    self.conf.register_opt(cfg.StrOpt('foo', required=True), group='blaa')
    paths = self.create_tempfiles([('test', '[blaa]\nfoo = bar')])
    self.conf(['--config-file', paths[0]])
    self.assertTrue(hasattr(self.conf, 'blaa'))
    self.assertTrue(hasattr(self.conf.blaa, 'foo'))
    self.assertEqual('bar', self.conf.blaa.foo)