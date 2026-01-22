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
def parseNestedParens(s, handleLiteral=1):
    """
    Parse an s-exp-like string into a more useful data structure.

    @type s: L{bytes}
    @param s: The s-exp-like string to parse

    @rtype: L{list} of L{bytes} and L{list}
    @return: A list containing the tokens present in the input.

    @raise MismatchedNesting: Raised if the number or placement
    of opening or closing parenthesis is invalid.
    """
    s = s.strip()
    inQuote = 0
    contentStack = [[]]
    try:
        i = 0
        L = len(s)
        while i < L:
            c = s[i:i + 1]
            if inQuote:
                if c == b'\\':
                    contentStack[-1].append(s[i:i + 2])
                    i += 2
                    continue
                elif c == b'"':
                    inQuote = not inQuote
                contentStack[-1].append(c)
                i += 1
            elif c == b'"':
                contentStack[-1].append(c)
                inQuote = not inQuote
                i += 1
            elif handleLiteral and c == b'{':
                end = s.find(b'}', i)
                if end == -1:
                    raise ValueError('Malformed literal')
                literalSize = int(s[i + 1:end])
                contentStack[-1].append((s[end + 3:end + 3 + literalSize],))
                i = end + 3 + literalSize
            elif c == b'(' or c == b'[':
                contentStack.append([])
                i += 1
            elif c == b')' or c == b']':
                contentStack[-2].append(contentStack.pop())
                i += 1
            else:
                contentStack[-1].append(c)
                i += 1
    except IndexError:
        raise MismatchedNesting(s)
    if len(contentStack) != 1:
        raise MismatchedNesting(s)
    return collapseStrings(contentStack[0])