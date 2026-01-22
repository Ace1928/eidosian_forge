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
def test_quotes_good(self):
    self.conf.register_cli_opt(cfg.StrOpt('foo', quotes=True))
    self.conf(['--foo', '"foobar1"'])
    self.assertEqual('foobar1', self.conf.foo)
    self.conf(['--foo', "'foobar2'"])
    self.assertEqual('foobar2', self.conf.foo)
    self.conf(['--foo', 'foobar3'])
    self.assertEqual('foobar3', self.conf.foo)
    self.conf(['--foo', 'foobar4"'])
    self.assertEqual('foobar4"', self.conf.foo)