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
def test_dict_sub_default_from_default_multi(self):
    self.conf.register_cli_opt(cfg.StrOpt('foo', default='floo'))
    self.conf.register_cli_opt(cfg.StrOpt('bar', default='blaa'))
    self.conf.register_cli_opt(cfg.StrOpt('goo', default='gloo'))
    self.conf.register_cli_opt(cfg.StrOpt('har', default='hlaa'))
    self.conf.register_cli_opt(cfg.DictOpt('dt', default={'$foo': '$bar', '$goo': 'goo', 'har': '$har', 'key1': 'str', 'key2': 12345}))
    self.conf([])
    self.assertEqual('blaa', self.conf.dt['floo'])
    self.assertEqual('goo', self.conf.dt['gloo'])
    self.assertEqual('hlaa', self.conf.dt['har'])
    self.assertEqual('str', self.conf.dt['key1'])
    self.assertEqual(12345, self.conf.dt['key2'])