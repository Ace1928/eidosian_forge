import argparse
import ctypes
import faulthandler
import hashlib
import io
import itertools
import logging
import multiprocessing
import os
import pickle
import random
import sys
import textwrap
import unittest
from collections import defaultdict
from contextlib import contextmanager
from importlib import import_module
from io import StringIO
import sqlparse
import django
from django.core.management import call_command
from django.db import connections
from django.test import SimpleTestCase, TestCase
from django.test.utils import NullTimeKeeper, TimeKeeper, iter_test_cases
from django.test.utils import setup_databases as _setup_databases
from django.test.utils import setup_test_environment
from django.test.utils import teardown_databases as _teardown_databases
from django.test.utils import teardown_test_environment
from django.utils.datastructures import OrderedSet
from django.utils.version import PY312
def load_tests_for_label(self, label, discover_kwargs):
    label_as_path = os.path.abspath(label)
    tests = None
    if not os.path.exists(label_as_path):
        with self.load_with_patterns():
            tests = self.test_loader.loadTestsFromName(label)
        if tests.countTestCases():
            return tests
    is_importable, is_package = try_importing(label)
    if is_importable:
        if not is_package:
            return tests
    elif not os.path.isdir(label_as_path):
        if os.path.exists(label_as_path):
            assert tests is None
            raise RuntimeError(f'One of the test labels is a path to a file: {label!r}, which is not supported. Use a dotted module name or path to a directory instead.')
        return tests
    kwargs = discover_kwargs.copy()
    if os.path.isdir(label_as_path) and (not self.top_level):
        kwargs['top_level_dir'] = find_top_level(label_as_path)
    with self.load_with_patterns():
        tests = self.test_loader.discover(start_dir=label, **kwargs)
    self.test_loader._top_level_dir = None
    return tests