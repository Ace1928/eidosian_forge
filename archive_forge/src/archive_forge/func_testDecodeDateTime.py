import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testDecodeDateTime(self):
    for datetime_string, datetime_vals in (('2012-09-30T15:31:50.262', (2012, 9, 30, 15, 31, 50, 262000)), ('2012-09-30T15:31:50', (2012, 9, 30, 15, 31, 50, 0))):
        message = protojson.decode_message(MyMessage, '{"a_datetime": "%s"}' % datetime_string)
        expected_message = MyMessage(a_datetime=datetime.datetime(*datetime_vals))
        self.assertEquals(expected_message, message)