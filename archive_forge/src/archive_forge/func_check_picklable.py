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
def check_picklable(self, test, err):
    try:
        self._confirm_picklable(err)
    except Exception as exc:
        original_exc_txt = repr(err[1])
        original_exc_txt = textwrap.fill(original_exc_txt, 75, initial_indent='    ', subsequent_indent='    ')
        pickle_exc_txt = repr(exc)
        pickle_exc_txt = textwrap.fill(pickle_exc_txt, 75, initial_indent='    ', subsequent_indent='    ')
        if tblib is None:
            print('\n\n{} failed:\n\n{}\n\nUnfortunately, tracebacks cannot be pickled, making it impossible for the\nparallel test runner to handle this exception cleanly.\n\nIn order to see the traceback, you should install tblib:\n\n    python -m pip install tblib\n'.format(test, original_exc_txt))
        else:
            print("\n\n{} failed:\n\n{}\n\nUnfortunately, the exception it raised cannot be pickled, making it impossible\nfor the parallel test runner to handle it cleanly.\n\nHere's the error encountered while trying to pickle the exception:\n\n{}\n\nYou should re-run this test with the --parallel=1 option to reproduce the\nfailure and get a correct traceback.\n".format(test, original_exc_txt, pickle_exc_txt))
        raise