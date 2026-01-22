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
def testRepeatedNested(self):
    """Test repeated nested messages."""
    nested_message1 = NestedMessage()
    nested_message1.a_value = u'a string'
    nested_message2 = NestedMessage()
    nested_message2.a_value = u'another string'
    message = HasNestedMessage()
    message.repeated_nested = [nested_message1, nested_message2]
    self.EncodeDecode(self.encoded_repeated_nested, message)