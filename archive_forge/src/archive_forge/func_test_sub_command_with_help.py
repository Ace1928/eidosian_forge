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
def test_sub_command_with_help(self):

    def add_parsers(subparsers):
        subparsers.add_parser('a')
    self.conf.register_cli_opt(cfg.SubCommandOpt('cmd', title='foo foo', description='bar bar', help='blaa blaa', handler=add_parsers))
    self.useFixture(fixtures.MonkeyPatch('sys.stdout', io.StringIO()))
    self.assertRaises(SystemExit, self.conf, ['--help'])
    self.assertIn('foo foo', sys.stdout.getvalue())
    self.assertIn('bar bar', sys.stdout.getvalue())
    self.assertIn('blaa blaa', sys.stdout.getvalue())