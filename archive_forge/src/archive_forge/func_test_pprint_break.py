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
def test_pprint_break():
    """
    Test that p.break_ produces expected output
    """
    output = pretty.pretty(Breaking())
    expected = 'TG: Breaking(\n    ):'
    assert output == expected