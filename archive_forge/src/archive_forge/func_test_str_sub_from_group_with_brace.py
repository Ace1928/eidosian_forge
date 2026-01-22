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
def test_str_sub_from_group_with_brace(self):
    self.conf.register_group(cfg.OptGroup('f'))
    self.conf.register_cli_opt(cfg.StrOpt('oo', default='blaa'), group='f')
    self.conf.register_cli_opt(cfg.StrOpt('bar', default='${f.oo}'))
    self.conf([])
    self.assertTrue(hasattr(self.conf, 'bar'))
    self.assertEqual('blaa', self.conf.bar)