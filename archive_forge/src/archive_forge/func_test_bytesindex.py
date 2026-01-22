import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_bytesindex():
    assert six.indexbytes(six.b('hello'), 3) == ord('l')