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
def test_deprecated_with_dest(self):
    self.conf.register_group(cfg.OptGroup('other'))
    self.conf.register_opt(cfg.StrOpt('foo-bar', deprecated_name='bar', dest='foo'), group='other')
    content = 'bar=baz'
    paths = self.create_tempfiles([('test', '[other]\n' + content + '\n')])
    self.conf(['--config-file', paths[0]])
    self.assertEqual('baz', self.conf.other.foo)
    expected = cfg._Namespace._deprecated_opt_message % {'dep_option': 'bar', 'dep_group': 'other', 'option': 'foo-bar', 'group': 'other'} + '\n'
    self.assertIn(expected, self.log_fixture.output)