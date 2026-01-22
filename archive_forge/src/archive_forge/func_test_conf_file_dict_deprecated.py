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
def test_conf_file_dict_deprecated(self):
    self.conf.register_opt(cfg.DictOpt('newfoo', deprecated_name='oldfoo'))
    paths = self.create_tempfiles([('test', '[DEFAULT]\noldfoo= key1:bar1\noldfoo = key2:bar2\n')])
    self.conf(['--config-file', paths[0]])
    self.assertTrue(hasattr(self.conf, 'newfoo'))
    self.assertEqual({'key2': 'bar2'}, self.conf.newfoo)