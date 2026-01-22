import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_add_doc():

    def f():
        """Icky doc"""
        pass
    six._add_doc(f, 'New doc')
    assert f.__doc__ == 'New doc'