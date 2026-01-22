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
def test_conf_file_int_string_default_type(self):
    self.conf.register_opt(cfg.IntOpt('foo', default='666'))
    self.conf([])
    self.assertEqual(666, self.conf.foo)