import datetime
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
def testValueFromMessage(self):
    message = message_types.DateTimeMessage(milliseconds=1991128000000)
    field = message_types.DateTimeField(1)
    timestamp = field.value_from_message(message)
    self.assertEqual(datetime.datetime(2033, 2, 4, 11, 6, 40), timestamp)