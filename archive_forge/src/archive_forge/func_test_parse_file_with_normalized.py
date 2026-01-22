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
def test_parse_file_with_normalized(self):
    paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = bar\n[BLAA]\nbar = foo\n')])
    sections = {}
    normalized = {}
    parser = cfg.ConfigParser(paths[0], sections)
    parser._add_normalized(normalized)
    parser.parse()
    self.assertIn('DEFAULT', sections)
    self.assertIn('DEFAULT', normalized)
    self.assertIn('BLAA', sections)
    self.assertIn('blaa', normalized)
    self.assertEqual(sections['DEFAULT']['foo'], ['bar'])
    self.assertEqual(normalized['DEFAULT']['foo'], ['bar'])
    self.assertEqual(sections['BLAA']['bar'], ['foo'])
    self.assertEqual(normalized['blaa']['bar'], ['foo'])