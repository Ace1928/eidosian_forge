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
def test_use_deprecated_for_removal_without_reason(self):
    self.conf.register_cli_opt(cfg.StrOpt('oldfoo', deprecated_for_removal=True))
    paths = self.create_tempfiles([('0', '[DEFAULT]\noldfoo = middle\n')])
    self.conf(['--oldfoo', 'first', '--config-file', paths[0]])
    log_out = self.logger.output
    self.assertIn('deprecated for removal.', log_out)