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
def test_hooks_invoked_once(self):
    fresh = {}
    result = [0]

    def foo(conf, foo_fresh):
        self.assertEqual(conf, self.conf)
        self.assertEqual(fresh, foo_fresh)
        result[0] += 1
    self.conf.register_mutate_hook(foo)
    self.conf.register_mutate_hook(foo)
    self._test_conf_files_mutate()
    self.assertEqual(1, result[0])