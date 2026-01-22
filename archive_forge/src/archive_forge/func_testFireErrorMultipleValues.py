from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import test_components as tc
from fire import testutils
from fire import trace
import mock
import six
def testFireErrorMultipleValues(self):
    error = core.FireError('Example error', 'value')
    self.assertIsNotNone(error)