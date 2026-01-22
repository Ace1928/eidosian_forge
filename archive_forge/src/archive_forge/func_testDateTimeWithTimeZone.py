import cgi
import datetime
import inspect
import os
import re
import socket
import types
import unittest
import six
from six.moves import range  # pylint: disable=redefined-builtin
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
def testDateTimeWithTimeZone(self):
    """Test DateTimeFields with time zones."""

    class MyMessage(messages.Message):
        value = message_types.DateTimeField(1)
    value = datetime.datetime(2013, 1, 3, 11, 36, 30, 123000, util.TimeZoneOffset(8 * 60))
    message = MyMessage(value=value)
    decoded = self.PROTOLIB.decode_message(MyMessage, self.PROTOLIB.encode_message(message))
    self.assertEquals(decoded.value, value)