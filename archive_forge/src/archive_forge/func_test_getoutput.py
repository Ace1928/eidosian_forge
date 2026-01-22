import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_getoutput():
    from six.moves import getoutput
    output = getoutput('echo "foo"')
    assert output == 'foo'