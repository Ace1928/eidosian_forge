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
def test_log_opt_values_from_sys_argv(self):
    self.useFixture(fixtures.MonkeyPatch('sys.argv', ['foo'] + self._args))
    self._do_test_log_opt_values(None)