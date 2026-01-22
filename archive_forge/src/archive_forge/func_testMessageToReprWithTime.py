import base64
import datetime
import json
import sys
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
def testMessageToReprWithTime(self):
    msg = TimeMessage(timefield=datetime.datetime(2014, 7, 2, 23, 33, 25, 541000, tzinfo=util.TimeZoneOffset(datetime.timedelta(0))))
    self.assertEqual(encoding.MessageToRepr(msg, multiline=True), '%s.TimeMessage(\n    timefield=datetime.datetime(2014, 7, 2, 23, 33, 25, 541000, tzinfo=apitools.base.protorpclite.util.TimeZoneOffset(datetime.timedelta(0))),\n)' % __name__)
    self.assertEqual(encoding.MessageToRepr(msg, multiline=True, no_modules=True), 'TimeMessage(\n    timefield=datetime.datetime(2014, 7, 2, 23, 33, 25, 541000, tzinfo=TimeZoneOffset(datetime.timedelta(0))),\n)')