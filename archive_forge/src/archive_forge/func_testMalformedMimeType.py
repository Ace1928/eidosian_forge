import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import util
def testMalformedMimeType(self):
    self.assertRaises(exceptions.InvalidUserInputError, util.AcceptableMimeType, ['*/*'], 'abcd')