import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_unichr():
    assert six.u('ሴ') == six.unichr(4660)
    assert type(six.u('ሴ')) is type(six.unichr(4660))