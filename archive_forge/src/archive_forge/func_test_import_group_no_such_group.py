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
def test_import_group_no_such_group(self):
    self.assertRaises(cfg.NoSuchGroupError, cfg.CONF.import_group, 'quxxx', 'oslo_config.tests.testmods.baz_qux_opt')