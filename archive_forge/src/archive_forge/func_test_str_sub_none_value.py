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
def test_str_sub_none_value(self):
    self.conf.register_cli_opt(cfg.StrOpt('oo'))
    self.conf.register_cli_opt(cfg.StrOpt('bar', default='$oo'))
    self.conf.register_cli_opt(cfg.StrOpt('barbar', default='foo $oo foo'))
    self.conf([])
    self.assertTrue(hasattr(self.conf, 'bar'))
    self.assertEqual('', self.conf.bar)
    self.assertEqual('foo  foo', self.conf.barbar)