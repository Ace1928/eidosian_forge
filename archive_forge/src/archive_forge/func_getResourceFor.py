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
def getResourceFor(self, request):
    """
        Get a resource for a request.

        This iterates through the resource hierarchy, calling
        getChildWithDefault on each resource it finds for a path element,
        stopping when it hits an element where isLeaf is true.
        """
    request.site = self
    request.sitepath = copy.copy(request.prepath)
    return resource.getChildForRequest(self.resource, request)