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
def test_arg_group_with_default(self):
    self.conf.register_group(cfg.OptGroup('blaa'))
    self.conf.register_cli_opt(cfg.StrOpt('foo', default='bar'), group='blaa')
    self.conf([])
    self.assertTrue(hasattr(self.conf, 'blaa'))
    self.assertTrue(hasattr(self.conf.blaa, 'foo'))
    self.assertEqual('bar', self.conf.blaa.foo)