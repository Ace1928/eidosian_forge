import binascii
import codecs
import copy
import email.utils
import functools
import re
import string
import tempfile
import time
import uuid
from base64 import decodebytes, encodebytes
from io import BytesIO
from itertools import chain
from typing import Any, List, cast
from zope.interface import implementer
from twisted.cred import credentials
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import defer, error, interfaces
from twisted.internet.defer import maybeDeferred
from twisted.mail._cred import (
from twisted.mail._except import (
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log, text
from twisted.python.compat import (
def fetchMessage(self, messages, uid=0):
    """
        Retrieve one or more entire messages

        This command is allowed in the Selected state.

        @type messages: L{MessageSet} or L{str}
        @param messages: A message sequence set

        @type uid: C{bool}
        @param uid: Indicates whether the message sequence set is of message
        numbers or of unique message IDs.

        @rtype: L{Deferred}

        @return: A L{Deferred} which will fire with a C{dict} mapping message
            sequence numbers to C{dict}s giving message data for the
            corresponding message.  If C{uid} is true, the inner dictionaries
            have a C{'UID'} key mapped to a L{str} giving the UID for the
            message.  The text of the message is a L{str} associated with the
            C{'RFC822'} key in each dictionary.
        """
    return self._fetch(messages, useUID=uid, rfc822=1)