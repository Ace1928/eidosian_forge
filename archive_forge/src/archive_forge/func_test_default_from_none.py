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
def test_default_from_none(self):
    opts = [cfg.StrOpt('foo')]
    self.conf.register_opts(opts)
    cfg.set_defaults(opts, foo='bar')
    self.conf([])
    self.assertEqual('bar', self.conf.foo)