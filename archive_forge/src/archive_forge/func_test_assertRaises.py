import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def test_assertRaises(self):

    def _raise(e):
        raise e
    self.assertRaises(KeyError, _raise, KeyError)
    self.assertRaises(KeyError, _raise, KeyError('key'))
    try:
        self.assertRaises(KeyError, lambda: None)
    except self.failureException as e:
        self.assertIn('KeyError not raised', str(e))
    else:
        self.fail("assertRaises() didn't fail")
    try:
        self.assertRaises(KeyError, _raise, ValueError)
    except ValueError:
        pass
    else:
        self.fail("assertRaises() didn't let exception pass through")
    with self.assertRaises(KeyError) as cm:
        try:
            raise KeyError
        except Exception as e:
            exc = e
            raise
    self.assertIs(cm.exception, exc)
    with self.assertRaises(KeyError):
        raise KeyError('key')
    try:
        with self.assertRaises(KeyError):
            pass
    except self.failureException as e:
        self.assertIn('KeyError not raised', str(e))
    else:
        self.fail("assertRaises() didn't fail")
    try:
        with self.assertRaises(KeyError):
            raise ValueError
    except ValueError:
        pass
    else:
        self.fail("assertRaises() didn't let exception pass through")