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
def test_cli(self):
    self.conf.register_cli_opt(cfg.StrOpt('foo'))
    key = (None, 'foo')
    self.assertAbsent(key)
    self.read('[DEFAULT]\nfoo = file0\n')
    self.assertValue(key, 'file0')
    self.read('[DEFAULT]\nfoo = file1\n')
    self.assertEqual('file1', self.ns._get_cli_value([key]))