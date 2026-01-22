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
def test_sub_default_from_default(self):
    self._prep_test_str_int_sub(foo_default='123', bar_default='$foo')
    self.conf([])
    self._assert_int_sub()