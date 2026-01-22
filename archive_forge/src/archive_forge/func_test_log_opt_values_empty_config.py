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
def test_log_opt_values_empty_config(self):
    empty_conf = cfg.ConfigOpts()
    logger = self.FakeLogger(self, 666)
    empty_conf.log_opt_values(logger, 666)
    self.assertEqual(['*' * 80, 'Configuration options gathered from:', 'command line args: None', 'config files: []', '=' * 80, 'config_source                  = []', '*' * 80], logger.logged)