import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testDecodeRepeatedDateTime(self):
    message = protojson.decode_message(MyMessage, '{"a_repeated_datetime": ["2012-09-30T15:31:50.262", "2010-01-21T09:52:00", "2000-01-01T01:00:59.999999"]}')
    expected_message = MyMessage(a_repeated_datetime=[datetime.datetime(2012, 9, 30, 15, 31, 50, 262000), datetime.datetime(2010, 1, 21, 9, 52), datetime.datetime(2000, 1, 1, 1, 0, 59, 999999)])
    self.assertEquals(expected_message, message)