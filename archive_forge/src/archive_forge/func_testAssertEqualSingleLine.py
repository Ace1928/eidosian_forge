import contextlib
import difflib
import pprint
import pickle
import re
import sys
import logging
import warnings
import weakref
import inspect
import types
from copy import deepcopy
from test import support
import unittest
from unittest.test.support import (
from test.support import captured_stderr, gc_collect
def testAssertEqualSingleLine(self):
    sample_text = 'laden swallows fly slowly'
    revised_sample_text = 'unladen swallows fly quickly'
    sample_text_error = '- laden swallows fly slowly\n?                    ^^^^\n+ unladen swallows fly quickly\n? ++                   ^^^^^\n'
    try:
        self.assertEqual(sample_text, revised_sample_text)
    except self.failureException as e:
        error = str(e).split('\n', 1)[1]
        self.assertEqual(sample_text_error, error)