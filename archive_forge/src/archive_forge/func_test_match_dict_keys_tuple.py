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
def test_match_dict_keys_tuple(self):
    """
        Test that match_dict_keys called with extra prefix works on a couple of use case,
        does return what expected, and does not crash.
        """
    delims = ' \t\n`!@#$^&*()=+[{]}\\|;:\'",<>?'
    keys = [('foo', 'bar'), ('foo', 'oof'), ('foo', b'bar'), ('other', 'test')]

    def match(*args, extra=None, **kwargs):
        quote, offset, matches = match_dict_keys(*args, delims=delims, extra_prefix=extra, **kwargs)
        return (quote, offset, list(matches))
    assert match(keys, "'", extra=('foo',)) == ("'", 1, ['bar', 'oof'])
    assert match(keys, '"', extra=('foo',)) == ('"', 1, ['bar', 'oof'])
    assert match(keys, "'o", extra=('foo',)) == ("'", 1, ['oof'])
    assert match(keys, '"o', extra=('foo',)) == ('"', 1, ['oof'])
    assert match(keys, "b'", extra=('foo',)) == ("'", 2, ['bar'])
    assert match(keys, 'b"', extra=('foo',)) == ('"', 2, ['bar'])
    assert match(keys, "b'b", extra=('foo',)) == ("'", 2, ['bar'])
    assert match(keys, 'b"b', extra=('foo',)) == ('"', 2, ['bar'])
    assert match(keys, "'", extra=('no_foo',)) == ("'", 1, [])
    assert match(keys, "'", extra=('fo',)) == ("'", 1, [])
    keys = [('foo1', 'foo2', 'foo3', 'foo4'), ('foo1', 'foo2', 'bar', 'foo4')]
    assert match(keys, "'foo", extra=('foo1',)) == ("'", 1, ['foo2'])
    assert match(keys, "'foo", extra=('foo1', 'foo2')) == ("'", 1, ['foo3'])
    assert match(keys, "'foo", extra=('foo1', 'foo2', 'foo3')) == ("'", 1, ['foo4'])
    assert match(keys, "'foo", extra=('foo1', 'foo2', 'foo3', 'foo4')) == ("'", 1, [])
    keys = [('foo', 1111), ('foo', '2222'), (3333, 'bar'), (3333, 4444)]
    assert match(keys, "'", extra=('foo',)) == ("'", 1, ['2222'])
    assert match(keys, '', extra=('foo',)) == ('', 0, ['1111', "'2222'"])
    assert match(keys, "'", extra=(3333,)) == ("'", 1, ['bar'])
    assert match(keys, '', extra=(3333,)) == ('', 0, ["'bar'", '4444'])
    assert match(keys, "'", extra=('3333',)) == ("'", 1, [])
    assert match(keys, '33') == ('', 0, ['3333'])