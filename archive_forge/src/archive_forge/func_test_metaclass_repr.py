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
def test_metaclass_repr():
    output = pretty.pretty(ClassWithMeta)
    assert output == '[CUSTOM REPR FOR CLASS ClassWithMeta]'