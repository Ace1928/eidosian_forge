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
def testSetSlice(self):
    field_list = messages.FieldList(self.integer_field, [1, 2, 3, 4, 5])
    field_list[1:3] = [10, 20]
    self.assertEquals([1, 10, 20, 4, 5], field_list)