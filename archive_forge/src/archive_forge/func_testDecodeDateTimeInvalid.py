import datetime
import sys
import types
import unittest
import six
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
def testDecodeDateTimeInvalid(self):
    """Test that decoding malformed datetime strings raises execptions."""
    for datetime_string in ('invalid', '2012-09-30T15:31:50.', '-08:00 2012-09-30T15:31:50.262', '2012-09-30T15:31', '2012-09-30T15:31Z', '2012-09-30T15:31:50ZZ', '2012-09-30T15:31:50.262 blah blah -08:00', '1000-99-99T25:99:99.999-99:99', '2012-09-30T15:31:50.262343123'):
        self.assertRaises(ValueError, util.decode_datetime, datetime_string)