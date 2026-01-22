import os
import pytest
import sys
import textwrap
import unittest
from contextlib import contextmanager
from traitlets.config.loader import Config
from IPython import get_ipython
from IPython.core import completer
from IPython.utils.tempdir import TemporaryDirectory, TemporaryWorkingDirectory
from IPython.utils.generics import complete_object
from IPython.testing import decorators as dec
from IPython.core.completer import (
def test_dict_key_completion_contexts(self):
    """Test expression contexts in which dict key completion occurs"""
    ip = get_ipython()
    complete = ip.Completer.complete
    d = {'abc': None}
    ip.user_ns['d'] = d

    class C:
        data = d
    ip.user_ns['C'] = C
    ip.user_ns['get'] = lambda: d
    ip.user_ns['nested'] = {'x': d}

    def assert_no_completion(**kwargs):
        _, matches = complete(**kwargs)
        self.assertNotIn('abc', matches)
        self.assertNotIn("abc'", matches)
        self.assertNotIn("abc']", matches)
        self.assertNotIn("'abc'", matches)
        self.assertNotIn("'abc']", matches)

    def assert_completion(**kwargs):
        _, matches = complete(**kwargs)
        self.assertIn("'abc'", matches)
        self.assertNotIn("'abc']", matches)
    assert_no_completion(line_buffer="d['a'")
    assert_no_completion(line_buffer='d["a"')
    assert_no_completion(line_buffer="d['a' + ")
    assert_no_completion(line_buffer="d['a' + '")
    assert_completion(line_buffer='+ d[')
    assert_completion(line_buffer='(d[')
    assert_completion(line_buffer='C.data[')
    assert_completion(line_buffer="nested['x'][")
    with evaluation_policy('minimal'):
        with pytest.raises(AssertionError):
            assert_completion(line_buffer="nested['x'][")

    def assert_completion(**kwargs):
        _, matches = complete(**kwargs)
        self.assertIn("get()['abc']", matches)
    assert_no_completion(line_buffer='get()[')
    with greedy_completion():
        assert_completion(line_buffer='get()[')
        assert_completion(line_buffer="get()['")
        assert_completion(line_buffer="get()['a")
        assert_completion(line_buffer="get()['ab")
        assert_completion(line_buffer="get()['abc")