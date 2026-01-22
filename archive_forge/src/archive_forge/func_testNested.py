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
def testNested(self):
    """Test nested messages."""
    nested_message = NestedMessage()
    nested_message.a_value = u'a string'
    message = HasNestedMessage()
    message.nested = nested_message
    self.EncodeDecode(self.encoded_nested, message)