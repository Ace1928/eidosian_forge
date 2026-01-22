import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_text_type():
    assert type(six.u('hi')) is six.text_type