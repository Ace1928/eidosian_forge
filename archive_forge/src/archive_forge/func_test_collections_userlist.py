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
def test_collections_userlist():
    a = UserList()
    a.append(a)
    cases = [(UserList(), 'UserList([])'), (UserList((i for i in range(1000, 1020))), 'UserList([1000,\n          1001,\n          1002,\n          1003,\n          1004,\n          1005,\n          1006,\n          1007,\n          1008,\n          1009,\n          1010,\n          1011,\n          1012,\n          1013,\n          1014,\n          1015,\n          1016,\n          1017,\n          1018,\n          1019])'), (a, 'UserList([UserList(...)])')]
    for obj, expected in cases:
        assert pretty.pretty(obj) == expected