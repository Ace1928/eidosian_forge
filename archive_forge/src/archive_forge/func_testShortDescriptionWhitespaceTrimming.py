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
@unittest.skipIf(sys.flags.optimize >= 2, 'Docstrings are omitted with -O2 and above')
def testShortDescriptionWhitespaceTrimming(self):
    """
            Tests shortDescription() whitespace is trimmed, so that the first
            line of nonwhite-space text becomes the docstring.
        """
    self.assertEqual(self.shortDescription(), 'Tests shortDescription() whitespace is trimmed, so that the first')