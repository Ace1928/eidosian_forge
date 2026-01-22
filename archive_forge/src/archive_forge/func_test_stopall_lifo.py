import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_stopall_lifo(self):
    stopped = []

    class thing(object):
        one = two = three = None

    def get_patch(attribute):

        class mypatch(_patch):

            def stop(self):
                stopped.append(attribute)
                return super(mypatch, self).stop()
        return mypatch(lambda: thing, attribute, None, None, False, None, None, None, {})
    [get_patch(val).start() for val in ('one', 'two', 'three')]
    patch.stopall()
    self.assertEqual(stopped, ['three', 'two', 'one'])