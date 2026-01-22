import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_moved_attribute(self):
    attr = six.MovedAttribute('spam', 'foo', 'bar')
    if six.PY3:
        assert attr.mod == 'bar'
    else:
        assert attr.mod == 'foo'
    assert attr.attr == 'spam'
    attr = six.MovedAttribute('spam', 'foo', 'bar', 'lemma')
    assert attr.attr == 'lemma'
    attr = six.MovedAttribute('spam', 'foo', 'bar', 'lemma', 'theorm')
    if six.PY3:
        assert attr.attr == 'theorm'
    else:
        assert attr.attr == 'lemma'