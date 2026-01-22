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
def test_pprint_break_repr():
    """
    Test that p.break_ is used in repr
    """
    output = pretty.pretty([[BreakingRepr()]])
    expected = '[[Breaking(\n  )]]'
    assert output == expected
    output = pretty.pretty([[BreakingRepr()] * 2])
    expected = '[[Breaking(\n  ),\n  Breaking(\n  )]]'
    assert output == expected