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
def testUnexpectedField(self):
    """Test decoding and encoding unexpected fields."""
    loaded_message = self.PROTOLIB.decode_message(OptionalMessage, self.unexpected_tag_message)
    self.assertEquals(OptionalMessage(), loaded_message)
    self.assertEquals(self.unexpected_tag_message, self.PROTOLIB.encode_message(loaded_message))