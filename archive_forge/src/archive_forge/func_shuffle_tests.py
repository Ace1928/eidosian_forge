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
def shuffle_tests(tests, shuffler):
    """
    Return an iterator over the given tests in a shuffled order, keeping tests
    next to other tests of their class.

    `tests` should be an iterable of tests.
    """
    tests_by_type = {}
    for _, class_tests in itertools.groupby(tests, type):
        class_tests = list(class_tests)
        test_type = type(class_tests[0])
        class_tests = shuffler.shuffle(class_tests, key=lambda test: test.id())
        tests_by_type[test_type] = class_tests
    classes = shuffler.shuffle(tests_by_type, key=_class_shuffle_key)
    return itertools.chain(*(tests_by_type[cls] for cls in classes))