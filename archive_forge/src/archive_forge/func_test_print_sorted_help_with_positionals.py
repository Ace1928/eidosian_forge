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
def test_print_sorted_help_with_positionals(self):
    f = io.StringIO()
    self.conf.register_cli_opt(cfg.StrOpt('pst', positional=True, required=False))
    self.conf.register_cli_opt(cfg.StrOpt('abc'))
    self.conf.register_cli_opt(cfg.StrOpt('zba'))
    self.conf.register_cli_opt(cfg.StrOpt('ghi'))
    self.conf([])
    self.conf.print_help(file=f)
    zba = f.getvalue().find('--zba')
    abc = f.getvalue().find('--abc')
    ghi = f.getvalue().find('--ghi')
    list = [abc, ghi, zba]
    self.assertEqual(sorted(list), list)