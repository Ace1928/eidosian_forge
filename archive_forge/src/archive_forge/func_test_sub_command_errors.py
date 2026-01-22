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
def test_sub_command_errors(self):

    def add_parsers(subparsers):
        sub = subparsers.add_parser('a')
        sub.add_argument('--bar')
    self.conf.register_cli_opt(cfg.BoolOpt('bar'))
    self.conf.register_cli_opt(cfg.SubCommandOpt('cmd', handler=add_parsers))
    self.conf(['a'])
    self.assertRaises(cfg.DuplicateOptError, getattr, self.conf.cmd, 'bar')
    self.assertRaises(cfg.NoSuchOptError, getattr, self.conf.cmd, 'foo')