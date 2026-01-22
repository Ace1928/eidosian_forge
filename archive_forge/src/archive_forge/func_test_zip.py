import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_zip():
    from six.moves import zip
    assert six.advance_iterator(zip(range(2), range(2))) == (0, 0)