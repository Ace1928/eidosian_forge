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
def test_required_opt_error_with_group(self):
    msg = str(cfg.RequiredOptError('foo', cfg.OptGroup('bar')))
    self.assertEqual('value required for option foo in group [bar]', msg)