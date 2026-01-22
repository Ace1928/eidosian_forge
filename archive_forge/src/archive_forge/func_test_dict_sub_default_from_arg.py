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
def test_dict_sub_default_from_arg(self):
    self.conf.register_cli_opt(cfg.StrOpt('foo', default=None))
    self.conf.register_cli_opt(cfg.StrOpt('bar', default=None))
    self.conf.register_cli_opt(cfg.DictOpt('dt', default={'$foo': '$bar'}))
    self.conf(['--foo', 'floo', '--bar', 'blaa'])
    self.assertTrue(hasattr(self.conf, 'dt'))
    self.assertEqual('blaa', self.conf.dt['floo'])