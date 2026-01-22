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
def testEncodeUninitialized(self):
    """Test that cannot encode uninitialized message."""
    required = NestedMessage()
    self.assertRaisesWithRegexpMatch(messages.ValidationError, 'Message NestedMessage is missing required field a_value', self.PROTOLIB.encode_message, required)