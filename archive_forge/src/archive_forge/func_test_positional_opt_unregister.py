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
def test_positional_opt_unregister(self):
    command = cfg.StrOpt('command', positional=True)
    arg1 = cfg.StrOpt('arg1', positional=True)
    arg2 = cfg.StrOpt('arg2', positional=True)
    self.conf.register_cli_opt(command)
    self.conf.register_cli_opt(arg1)
    self.conf.register_cli_opt(arg2)
    self.conf(['command', 'arg1', 'arg2'])
    self.assertEqual('command', self.conf.command)
    self.assertEqual('arg1', self.conf.arg1)
    self.assertEqual('arg2', self.conf.arg2)
    self.conf.reset()
    self.conf.unregister_opt(arg1)
    self.conf.unregister_opt(arg2)
    arg0 = cfg.StrOpt('arg0', positional=True)
    self.conf.register_cli_opt(arg0)
    self.conf.register_cli_opt(arg1)
    self.conf(['command', 'arg0', 'arg1'])
    self.assertEqual('command', self.conf.command)
    self.assertEqual('arg0', self.conf.arg0)
    self.assertEqual('arg1', self.conf.arg1)