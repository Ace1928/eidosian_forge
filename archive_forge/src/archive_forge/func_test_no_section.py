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
def test_no_section(self):
    with tempfile.NamedTemporaryFile() as tmpfile:
        tmpfile.write(b'foo = bar')
        tmpfile.flush()
        parser = cfg.ConfigParser(tmpfile.name, {})
        self.assertRaises(cfg.ParseError, parser.parse)