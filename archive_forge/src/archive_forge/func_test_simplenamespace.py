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
def test_simplenamespace():
    SN = types.SimpleNamespace
    sn_recursive = SN()
    sn_recursive.first = sn_recursive
    sn_recursive.second = sn_recursive
    cases = [(SN(), 'namespace()'), (SN(x=SN()), 'namespace(x=namespace())'), (SN(a_long_name=[SN(s=string.ascii_lowercase)] * 3, a_short_name=None), "namespace(a_long_name=[namespace(s='abcdefghijklmnopqrstuvwxyz'),\n                       namespace(s='abcdefghijklmnopqrstuvwxyz'),\n                       namespace(s='abcdefghijklmnopqrstuvwxyz')],\n          a_short_name=None)"), (sn_recursive, 'namespace(first=namespace(...), second=namespace(...))')]
    for obj, expected in cases:
        assert pretty.pretty(obj) == expected