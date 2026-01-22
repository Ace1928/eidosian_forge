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
def test_conf_file_float_only_max_in_range(self):
    self.conf.register_opt(cfg.FloatOpt('foo', max=5.5))
    paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 4.5\n')])
    self.conf(['--config-file', paths[0]])
    self.assertTrue(hasattr(self.conf, 'foo'))
    self.assertEqual(4.5, self.conf.foo)