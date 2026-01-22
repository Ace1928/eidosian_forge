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
def fetchEnvelope(self, messages, uid=0):
    """
        Retrieve the envelope data for one or more messages

        This command is allowed in the Selected state.

        @type messages: L{MessageSet} or L{str}
        @param messages: The messages for which to retrieve envelope
            data.

        @type uid: L{bool}
        @param uid: Indicates whether the message sequence set is of
            message numbers or of unique message IDs.

        @rtype: L{Deferred}
        @return: A deferred whose callback is invoked with a dict
            mapping message numbers to envelope data, or whose errback
            is invoked if there is an error.  Envelope data consists
            of a sequence of the date, subject, from, sender,
            reply-to, to, cc, bcc, in-reply-to, and message-id header
            fields.  The date, subject, in-reply-to, and message-id
            fields are L{str}, while the from, sender, reply-to, to,
            cc, and bcc fields contain address data as L{str}s.
            Address data consists of a sequence of name, source route,
            mailbox name, and hostname.  Fields which are not present
            for a particular address may be L{None}.
        """
    return self._fetch(messages, useUID=uid, envelope=1)