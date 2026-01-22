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
def test_unregister_opt(self):
    opts = [cfg.StrOpt('foo'), cfg.StrOpt('bar')]
    self.conf.register_opts(opts)
    self.assertTrue(hasattr(self.conf, 'foo'))
    self.assertTrue(hasattr(self.conf, 'bar'))
    self.conf.unregister_opt(opts[0])
    self.assertFalse(hasattr(self.conf, 'foo'))
    self.assertTrue(hasattr(self.conf, 'bar'))
    self.conf([])
    self.assertRaises(cfg.ArgsAlreadyParsedError, self.conf.unregister_opt, opts[1])
    self.conf.clear()
    self.assertTrue(hasattr(self.conf, 'bar'))
    self.conf.unregister_opts(opts)