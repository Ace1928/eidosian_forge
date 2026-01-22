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
def test_bad_cli_arg(self):
    self.conf.register_opt(cfg.BoolOpt('foo'))
    self.useFixture(fixtures.MonkeyPatch('sys.stderr', io.StringIO()))
    self.assertRaises(SystemExit, self.conf, ['--foo'])
    self.assertIn('error', sys.stderr.getvalue())
    self.assertIn('--foo', sys.stderr.getvalue())