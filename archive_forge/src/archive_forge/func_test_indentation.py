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
def test_indentation():
    """Test correct indentation in groups"""
    count = 40
    gotoutput = pretty.pretty(MyList(range(count)))
    expectedoutput = 'MyList(\n' + ',\n'.join(('   %d' % i for i in range(count))) + ')'
    assert gotoutput == expectedoutput