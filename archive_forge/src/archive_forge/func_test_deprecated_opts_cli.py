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
def test_deprecated_opts_cli(self):
    oldopts = [cfg.DeprecatedOpt(self.opt_name)]
    self.conf.register_cli_opt(cfg.BoolOpt('foobar', deprecated_opts=oldopts))
    self.conf(['--' + self.cli_name])
    self.assertTrue(self.conf.foobar)