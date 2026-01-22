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
def test_set_override_not_in_choices(self):
    self.conf.register_group(cfg.OptGroup('f'))
    self.conf.register_cli_opt(cfg.StrOpt('oo', choices=('a', 'b')), group='f')
    self.assertRaises(ValueError, self.conf.set_override, 'oo', 'c', 'f')