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
def testDecodeInvalidEnumType(self):
    decoded = self.PROTOLIB.decode_message(OptionalMessage, self.encoded_invalid_enum)
    message = OptionalMessage()
    self.assertEqual(message, decoded)
    encoded = self.PROTOLIB.encode_message(decoded)
    self.assertEqual(self.encoded_invalid_enum, encoded)