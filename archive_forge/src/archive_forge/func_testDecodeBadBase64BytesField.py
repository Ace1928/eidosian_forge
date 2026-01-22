import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testDecodeBadBase64BytesField(self):
    """Test decoding improperly encoded base64 bytes value."""
    self.assertRaisesWithRegexpMatch(messages.DecodeError, 'Base64 decoding error: Incorrect padding', protojson.decode_message, test_util.OptionalMessage, '{"bytes_value": "abcdefghijklmnopq"}')