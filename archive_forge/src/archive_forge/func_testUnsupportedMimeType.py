import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import util
def testUnsupportedMimeType(self):
    self.assertRaises(exceptions.GeneratedClientError, util.AcceptableMimeType, ['text/html;q=0.9'], 'text/html')