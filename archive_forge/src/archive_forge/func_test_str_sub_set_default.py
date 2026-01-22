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
def test_str_sub_set_default(self):
    self._prep_test_str_sub()
    self.conf.set_default('bar', '$foo')
    self.conf.set_default('foo', 'blaa')
    self.conf([])
    self._assert_str_sub()