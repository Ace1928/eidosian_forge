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
def test_config_file_opt_name(self):
    self.conf.register_opt(cfg.BoolOpt(self.opt_name))
    paths = self.create_tempfiles([('test', '[DEFAULT]\n' + self.cf_name + ' = True\n' + self.broken_cf_name + ' = False\n')])
    self.conf(['--config-file', paths[0]])
    self.assertTrue(getattr(self.conf, self.opt_dest))