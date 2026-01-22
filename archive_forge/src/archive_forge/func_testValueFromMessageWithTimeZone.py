import datetime
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
def testValueFromMessageWithTimeZone(self):
    message = message_types.DateTimeMessage(milliseconds=1991128000000, time_zone_offset=300)
    field = message_types.DateTimeField(1)
    timestamp = field.value_from_message(message)
    time_zone = util.TimeZoneOffset(60 * 5)
    self.assertEqual(datetime.datetime(2033, 2, 4, 11, 6, 40, tzinfo=time_zone), timestamp)