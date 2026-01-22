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
def reorder_tests(tests, classes, reverse=False, shuffler=None):
    """
    Reorder an iterable of tests, grouping by the given TestCase classes.

    This function also removes any duplicates and reorders so that tests of the
    same type are consecutive.

    The result is returned as an iterator. `classes` is a sequence of types.
    Tests that are instances of `classes[0]` are grouped first, followed by
    instances of `classes[1]`, etc. Tests that are not instances of any of the
    classes are grouped last.

    If `reverse` is True, the tests within each `classes` group are reversed,
    but without reversing the order of `classes` itself.

    The `shuffler` argument is an optional instance of this module's `Shuffler`
    class. If provided, tests will be shuffled within each `classes` group, but
    keeping tests with other tests of their TestCase class. Reversing is
    applied after shuffling to allow reversing the same random order.
    """
    bins = [defaultdict(OrderedSet) for i in range(len(classes) + 1)]
    *class_bins, last_bin = bins
    for test in tests:
        for test_bin, test_class in zip(class_bins, classes):
            if isinstance(test, test_class):
                break
        else:
            test_bin = last_bin
        test_bin[type(test)].add(test)
    for test_bin in bins:
        tests = list(itertools.chain.from_iterable(test_bin.values()))
        yield from reorder_test_bin(tests, shuffler=shuffler, reverse=reverse)