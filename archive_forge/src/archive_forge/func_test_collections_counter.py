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
def test_collections_counter():

    class MyCounter(Counter):
        pass
    cases = [(Counter(), 'Counter()'), (Counter(a=1), "Counter({'a': 1})"), (MyCounter(a=1), "MyCounter({'a': 1})"), (Counter(a=1, c=22), "Counter({'c': 22, 'a': 1})")]
    for obj, expected in cases:
        assert pretty.pretty(obj) == expected