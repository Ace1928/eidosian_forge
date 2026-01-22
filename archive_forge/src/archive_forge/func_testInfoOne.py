from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import unittest
from fire import inspectutils
from fire import test_components as tc
from fire import testutils
import six
def testInfoOne(self):
    info = inspectutils.Info(1)
    self.assertEqual(info.get('type_name'), 'int')
    self.assertEqual(info.get('file'), None)
    self.assertEqual(info.get('line'), None)
    self.assertEqual(info.get('string_form'), '1')