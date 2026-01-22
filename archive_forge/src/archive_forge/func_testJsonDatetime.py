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
def testJsonDatetime(self):
    msg = TimeMessage(timefield=datetime.datetime(2014, 7, 2, 23, 33, 25, 541000, tzinfo=util.TimeZoneOffset(datetime.timedelta(0))))
    self.assertEqual('{"timefield": "2014-07-02T23:33:25.541000+00:00"}', encoding.MessageToJson(msg))