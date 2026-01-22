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
def test_warn_immutability(self):
    self.log_fixture = self.useFixture(fixtures.FakeLogger())
    self.conf.register_cli_opt(cfg.StrOpt('foo', mutable=True))
    self.conf.register_cli_opt(cfg.StrOpt('boo'), group=self.my_group)
    self._test_conf_files_mutate()
    self.assertEqual('Ignoring change to immutable option group.boo\nOption DEFAULT.foo changed from [old_foo] to [new_foo]\n', self.log_fixture.output)