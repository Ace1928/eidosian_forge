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
def test_conf_files_mutate_none(self):
    """Test that immutable opts are not reloaded"""
    self.conf.register_cli_opt(cfg.StrOpt('foo'))
    self._test_conf_files_mutate()
    self.assertTrue(hasattr(self.conf, 'foo'))
    self.assertEqual('old_foo', self.conf.foo)