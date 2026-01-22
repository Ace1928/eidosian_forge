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
def test_int_override_with_None(self):
    self.conf.register_opt(cfg.IntOpt('foo'))
    self.conf.set_override('foo', None)
    self.conf([])
    self.assertIsNone(self.conf.foo)
    self.conf.clear_override('foo')
    self.assertIsNone(self.conf.foo)