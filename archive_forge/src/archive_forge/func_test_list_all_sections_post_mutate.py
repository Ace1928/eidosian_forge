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
def test_list_all_sections_post_mutate(self):
    paths = self.create_tempfiles([('test.ini', '[DEFAULT]\nfoo = bar\n[BLAA]\nbar = foo\n'), ('test2.ini', '[WOMBAT]\nwoo = war\n[BLAA]\nbar = foo\n')])
    self.conf(args=[], default_config_files=paths[:1])
    self.assertEqual(['BLAA', 'DEFAULT'], self.conf.list_all_sections())
    shutil.copy(paths[1], paths[0])
    self.conf.mutate_config_files()
    self.assertEqual(['BLAA', 'DEFAULT', 'WOMBAT'], self.conf.list_all_sections())