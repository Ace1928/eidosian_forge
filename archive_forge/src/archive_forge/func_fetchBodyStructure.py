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
def fetchBodyStructure(self, messages, uid=0):
    """
        Retrieve the structure of the body of one or more messages

        This command is allowed in the Selected state.

        @type messages: L{MessageSet} or L{str}
        @param messages: The messages for which to retrieve body structure
        data.

        @type uid: L{bool}
        @param uid: Indicates whether the message sequence set is of message
        numbers or of unique message IDs.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with a dict mapping
        message numbers to body structure data, or whose errback is invoked
        if there is an error.  Body structure data describes the MIME-IMB
        format of a message and consists of a sequence of mime type, mime
        subtype, parameters, content id, description, encoding, and size.
        The fields following the size field are variable: if the mime
        type/subtype is message/rfc822, the contained message's envelope
        information, body structure data, and number of lines of text; if
        the mime type is text, the number of lines of text.  Extension fields
        may also be included; if present, they are: the MD5 hash of the body,
        body disposition, body language.
        """
    return self._fetch(messages, useUID=uid, bodystructure=1)