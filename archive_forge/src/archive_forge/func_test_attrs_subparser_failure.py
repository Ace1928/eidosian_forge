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
def test_attrs_subparser_failure(self):
    CONF = cfg.ConfigOpts()
    CONF.register_cli_opt(cfg.SubCommandOpt('foo', handler=lambda sub: sub.add_parser('foo')))
    self.assertRaises(SystemExit, CONF, ['foo', 'bar'])