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
def test_str_sub_default_from_config_file(self):
    self._prep_test_str_sub(bar_default='$foo')
    paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = blaa\n')])
    self.conf(['--config-file', paths[0]])
    self._assert_str_sub()