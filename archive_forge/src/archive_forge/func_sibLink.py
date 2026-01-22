import copy
import os
import re
import zlib
from binascii import hexlify
from html import escape
from typing import List, Optional
from urllib.parse import quote as _quote
from zope.interface import implementer
from incremental import Version
from twisted import copyright
from twisted.internet import address, interfaces
from twisted.internet.error import AlreadyCalled, AlreadyCancelled
from twisted.logger import Logger
from twisted.python import components, failure, reflect
from twisted.python.compat import nativeString, networkString
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.spread.pb import Copyable, ViewPoint
from twisted.web import http, iweb, resource, util
from twisted.web.error import UnsupportedMethod
from twisted.web.http import unquote
def sibLink(self, name):
    """
        Return the text that links to a sibling of the requested resource.

        @param name: The sibling resource
        @type name: C{bytes}

        @return: A relative URL.
        @rtype: C{bytes}
        """
    if self.postpath:
        return len(self.postpath) * b'../' + name
    else:
        return name