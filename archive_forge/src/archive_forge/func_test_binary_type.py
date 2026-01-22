import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_binary_type():
    assert type(six.b('hi')) is six.binary_type