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
def test_required_cli_opt(self):
    self.conf.register_cli_opt(cfg.StrOpt('foo', required=True))
    self.conf(['--foo', 'bar'])
    self.assertTrue(hasattr(self.conf, 'foo'))
    self.assertEqual('bar', self.conf.foo)