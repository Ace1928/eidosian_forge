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
def test_sub_command_with_dest(self):

    def add_parsers(subparsers):
        subparsers.add_parser('a')
    self.conf.register_cli_opt(cfg.SubCommandOpt('cmd', dest='command', handler=add_parsers))
    self.assertTrue(hasattr(self.conf, 'command'))
    self.conf(['a'])
    self.assertEqual('a', self.conf.command.name)