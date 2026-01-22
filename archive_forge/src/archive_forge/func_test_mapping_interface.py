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
def test_mapping_interface(self):
    self.conf.register_cli_opt(cfg.StrOpt('foo'))
    self.conf(['--foo', 'bar'])
    self.assertIn('foo', self.conf)
    self.assertIn('config_file', self.conf)
    self.assertEqual(len(self.conf), 4)
    self.assertEqual('bar', self.conf['foo'])
    self.assertEqual('bar', self.conf.get('foo'))
    self.assertIn('bar', list(self.conf.values()))