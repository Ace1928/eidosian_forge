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
def removeFlags(self, messages, flags, silent=1, uid=0):
    """
        Remove from the set flags for one or more messages.

        This command is allowed in the Selected state.

        @type messages: L{MessageSet} or L{str}
        @param messages: A message sequence set

        @type flags: Any iterable of L{str}
        @param flags: The flags to set

        @type silent: L{bool}
        @param silent: If true, cause the server to suppress its verbose
        response.

        @type uid: L{bool}
        @param uid: Indicates whether the message sequence set is of message
        numbers or of unique message IDs.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with a list of the
        server's responses (C{[]} if C{silent} is true) or whose
        errback is invoked if there is an error.
        """
    return self._store(messages, b'-FLAGS', silent, flags, uid)