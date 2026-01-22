import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testConvertIntegerToFloat(self):
    """Test that integers passed in to float fields are converted.

        This is necessary because JSON outputs integers for numbers
        with 0 decimals.

        """
    message = protojson.decode_message(MyMessage, '{"a_float": 10}')
    self.assertTrue(isinstance(message.a_float, float))
    self.assertEquals(10.0, message.a_float)