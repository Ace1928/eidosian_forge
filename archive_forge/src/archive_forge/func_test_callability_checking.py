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
def test_callability_checking():
    """
    Test that the _repr_pretty_ method is tested for callability and skipped if
    not.
    """
    gotoutput = pretty.pretty(Dummy2())
    expectedoutput = 'Dummy1(...)'
    assert gotoutput == expectedoutput