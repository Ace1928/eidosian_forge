import copy
import pickle
import sys
import tempfile
import six
import unittest2 as unittest
import mock
from mock import (
from mock.mock import _CallList
from mock.tests.support import (
def test_assert_has_calls(self):
    kalls1 = [call(1, 2), ({'a': 3},), ((3, 4),), call(b=6), ('', (1,), {'b': 6})]
    kalls2 = [call.foo(), call.bar(1)]
    kalls2.extend(call.spam().baz(a=3).call_list())
    kalls2.extend(call.bam(set(), foo={}).fish([1]).call_list())
    mocks = []
    for mock in (Mock(), MagicMock()):
        mock(1, 2)
        mock(a=3)
        mock(3, 4)
        mock(b=6)
        mock(1, b=6)
        mocks.append((mock, kalls1))
    mock = Mock()
    mock.foo()
    mock.bar(1)
    mock.spam().baz(a=3)
    mock.bam(set(), foo={}).fish([1])
    mocks.append((mock, kalls2))
    for mock, kalls in mocks:
        for i in range(len(kalls)):
            for step in (1, 2, 3):
                these = kalls[i:i + step]
                mock.assert_has_calls(these)
                if len(these) > 1:
                    self.assertRaises(AssertionError, mock.assert_has_calls, list(reversed(these)))