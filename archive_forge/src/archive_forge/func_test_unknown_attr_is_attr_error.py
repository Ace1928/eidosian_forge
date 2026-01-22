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
def test_unknown_attr_is_attr_error(self):
    self.conf([])
    self.assertFalse(hasattr(self.conf, 'foo'))
    self.assertRaises(AttributeError, getattr, self.conf, 'foo')