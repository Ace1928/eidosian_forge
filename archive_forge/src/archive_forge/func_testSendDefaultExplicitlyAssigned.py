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
def testSendDefaultExplicitlyAssigned(self):
    """Test that default is sent when explcitly assigned."""
    message = HasDefault()
    message.a_value = HasDefault.a_value.default
    self.EncodeDecode(self.encoded_default_assigned, message)