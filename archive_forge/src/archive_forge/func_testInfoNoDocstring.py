from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import unittest
from fire import inspectutils
from fire import test_components as tc
from fire import testutils
import six
def testInfoNoDocstring(self):
    info = inspectutils.Info(tc.NoDefaults)
    self.assertEqual(info['docstring'], None, 'Docstring should be None')