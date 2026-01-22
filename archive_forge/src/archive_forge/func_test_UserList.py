import array
import unittest
from collections import OrderedDict
from collections import abc
from collections import deque
from types import MappingProxyType
from zope.interface import Invalid
from zope.interface._compat import PYPY
from zope.interface.common import collections
from . import VerifyClassMixin
from . import VerifyObjectMixin
from . import add_abc_interface_tests
def test_UserList(self):
    self.assertTrue(self.verify(collections.IMutableSequence, collections.UserList))