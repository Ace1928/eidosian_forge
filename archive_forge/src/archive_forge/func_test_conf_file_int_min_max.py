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
def test_conf_file_int_min_max(self):
    self.conf.register_opt(cfg.IntOpt('foo', min=1, max=5))
    paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 10\n')])
    self.conf(['--config-file', paths[0]])
    self.assertRaises(cfg.ConfigFileValueError, self.conf._get, 'foo')