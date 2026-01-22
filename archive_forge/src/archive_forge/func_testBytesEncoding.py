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
def testBytesEncoding(self):
    b64_str = 'AAc+'
    b64_msg = '{"field": "%s"}' % b64_str
    urlsafe_b64_str = 'AAc-'
    urlsafe_b64_msg = '{"field": "%s"}' % urlsafe_b64_str
    data = base64.b64decode(b64_str)
    msg = BytesMessage(field=data)
    self.assertEqual(msg, encoding.JsonToMessage(BytesMessage, urlsafe_b64_msg))
    self.assertEqual(msg, encoding.JsonToMessage(BytesMessage, b64_msg))
    self.assertEqual(urlsafe_b64_msg, encoding.MessageToJson(msg))
    enc_rep_msg = '{"repfield": ["%(b)s", "%(b)s"]}' % {'b': urlsafe_b64_str}
    rep_msg = BytesMessage(repfield=[data, data])
    self.assertEqual(rep_msg, encoding.JsonToMessage(BytesMessage, enc_rep_msg))
    self.assertEqual(enc_rep_msg, encoding.MessageToJson(rep_msg))