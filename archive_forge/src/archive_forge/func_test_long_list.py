from collections import Counter, defaultdict, deque, OrderedDict, UserList
import os
import pytest
import types
import string
import sys
import unittest
import pytest
from IPython.lib import pretty
from io import StringIO
def test_long_list(self):
    lis = list(range(10000))
    p = pretty.pretty(lis)
    last2 = p.rsplit('\n', 2)[-2:]
    self.assertEqual(last2, [' 999,', ' ...]'])