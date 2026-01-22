import pickle
import re
import sys
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testConstructor_InvalidValues(self):
    self.assertRaisesWithRegexpMatch(messages.ValidationError, re.escape('Expected type %r for IntegerField, found 1 (type %r)' % (six.integer_types, str)), messages.FieldList, self.integer_field, ['1', '2', '3'])