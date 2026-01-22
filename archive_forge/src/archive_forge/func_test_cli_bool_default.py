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
def test_cli_bool_default(self):
    self.conf.register_cli_opt(cfg.BoolOpt('foo'))
    self.conf.set_default('foo', True)
    self.assertTrue(self.conf.foo)
    self.conf([])
    self.assertTrue(self.conf.foo)
    self.conf.set_default('foo', False)
    self.assertFalse(self.conf.foo)
    self.conf.clear_default('foo')
    self.assertIsNone(self.conf.foo)