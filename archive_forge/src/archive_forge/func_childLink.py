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
def childLink(self, name):
    """
        Return the text that links to a child of the requested resource.

        @param name: The child resource
        @type name: C{bytes}

        @return: A relative URL.
        @rtype: C{bytes}
        """
    lpp = len(self.postpath)
    if lpp > 1:
        return (lpp - 1) * b'../' + name
    elif lpp == 1:
        return name
    elif len(self.prepath) and self.prepath[-1]:
        return self.prepath[-1] + b'/' + name
    else:
        return name