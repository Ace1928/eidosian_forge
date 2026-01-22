from __future__ import absolute_import, division, print_function
import collections
import functools
import sys
import unittest2 as unittest
import funcsigs as inspect
def test_signature_on_builtin_function(self):
    with self.assertRaisesRegex(ValueError, 'not supported by signature'):
        inspect.signature(type)
    with self.assertRaisesRegex(ValueError, 'not supported by signature'):
        inspect.signature(type.__call__)
        if hasattr(sys, 'pypy_version_info'):
            raise ValueError('not supported by signature')
    with self.assertRaisesRegex(ValueError, 'not supported by signature'):
        inspect.signature(min.__call__)
        if hasattr(sys, 'pypy_version_info'):
            raise ValueError('not supported by signature')
    with self.assertRaisesRegex(ValueError, 'no signature found for builtin function'):
        inspect.signature(min)