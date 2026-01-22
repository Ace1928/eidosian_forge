from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import unittest
from fire import inspectutils
from fire import test_components as tc
from fire import testutils
import six
def testGetFullArgSpecFromBuiltin(self):
    spec = inspectutils.GetFullArgSpec('test'.upper)
    self.assertEqual(spec.args, [])
    self.assertEqual(spec.defaults, ())
    self.assertEqual(spec.kwonlyargs, [])
    self.assertEqual(spec.kwonlydefaults, {})
    self.assertEqual(spec.annotations, {})