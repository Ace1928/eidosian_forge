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
def test_conf_file_dict_value_override(self):
    self.conf.register_cli_opt(cfg.DictOpt('foo'))
    paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = key:bar,key2:bar\n'), ('2', '[DEFAULT]\nfoo = k1:v1,k2:v2\n')])
    self.conf(['--foo', 'x:y,x2:y2', '--config-file', paths[0], '--config-file', paths[1]])
    self.assertTrue(hasattr(self.conf, 'foo'))
    self.assertEqual({'k1': 'v1', 'k2': 'v2'}, self.conf.foo)