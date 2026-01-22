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
def test_pprint_nomod():
    """
    Test that pprint works for classes with no __module__.
    """
    output = pretty.pretty(NoModule)
    assert output == 'NoModule'