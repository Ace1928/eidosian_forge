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
def test_do_not_use_default_single_arg(self):
    self.conf.register_opt(cfg.StrOpt('foo'))
    paths = self.create_tempfiles([('foo.conf.d/foo', '[DEFAULT]\nfoo = bar\n')])
    p = os.path.dirname(paths[0])
    self.conf(args=['--config-dir=' + p], default_config_dirs=['bar.conf.d'])
    self.assertEqual([p], self.conf.config_dir)
    self.assertEqual('bar', self.conf.foo)