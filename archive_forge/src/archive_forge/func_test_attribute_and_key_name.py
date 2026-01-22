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
def test_attribute_and_key_name(self):
    self.conf.register_opt(cfg.StrOpt(self.opt_name))
    self.assertTrue(hasattr(self.conf, self.opt_dest))
    self.assertFalse(hasattr(self.conf, self.broken_opt_dest))
    self.assertIn(self.opt_dest, self.conf)
    self.assertNotIn(self.broken_opt_dest, self.conf)