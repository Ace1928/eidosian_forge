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
def reorder_test_bin(tests, shuffler=None, reverse=False):
    """
    Return an iterator that reorders the given tests, keeping tests next to
    other tests of their class.

    `tests` should be an iterable of tests that supports reversed().
    """
    if shuffler is None:
        if reverse:
            return reversed(tests)
        return iter(tests)
    tests = shuffle_tests(tests, shuffler)
    if not reverse:
        return tests
    return reversed(list(tests))