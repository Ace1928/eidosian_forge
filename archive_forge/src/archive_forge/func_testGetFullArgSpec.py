from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import unittest
from fire import inspectutils
from fire import test_components as tc
from fire import testutils
import six
def testGetFullArgSpec(self):
    spec = inspectutils.GetFullArgSpec(tc.identity)
    self.assertEqual(spec.args, ['arg1', 'arg2', 'arg3', 'arg4'])
    self.assertEqual(spec.defaults, (10, 20))
    self.assertEqual(spec.varargs, 'arg5')
    self.assertEqual(spec.varkw, 'arg6')
    self.assertEqual(spec.kwonlyargs, [])
    self.assertEqual(spec.kwonlydefaults, {})
    self.assertEqual(spec.annotations, {'arg2': int, 'arg4': int})