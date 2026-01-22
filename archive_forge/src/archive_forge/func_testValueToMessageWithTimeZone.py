import datetime
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
def testValueToMessageWithTimeZone(self):
    time_zone = util.TimeZoneOffset(60 * 10)
    field = message_types.DateTimeField(1)
    message = field.value_to_message(datetime.datetime(2033, 2, 4, 11, 22, 10, tzinfo=time_zone))
    self.assertEqual(message_types.DateTimeMessage(milliseconds=1991128930000, time_zone_offset=600), message)