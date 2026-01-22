from __future__ import print_function
import gc
import sys
import unittest
from functools import partial
from unittest import skipUnless
from unittest import skipIf
from greenlet import greenlet
from greenlet import getcurrent
from . import TestCase
def test_contextvars_errors(self):
    let1 = greenlet(getcurrent().switch)
    self.assertFalse(hasattr(let1, 'gr_context'))
    with self.assertRaises(AttributeError):
        getattr(let1, 'gr_context')
    with self.assertRaises(AttributeError):
        let1.gr_context = None
    let1.switch()
    with self.assertRaises(AttributeError):
        getattr(let1, 'gr_context')
    with self.assertRaises(AttributeError):
        let1.gr_context = None
    del let1