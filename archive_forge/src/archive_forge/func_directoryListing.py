from __future__ import annotations
import errno
import itertools
import mimetypes
import os
import time
import warnings
from html import escape
from typing import Any, Callable, Dict, Sequence
from urllib.parse import quote, unquote
from zope.interface import implementer
from incremental import Version
from typing_extensions import Literal
from twisted.internet import abstract, interfaces
from twisted.python import components, filepath, log
from twisted.python.compat import nativeString, networkString
from twisted.python.deprecate import deprecated
from twisted.python.runtime import platformType
from twisted.python.url import URL
from twisted.python.util import InsensitiveDict
from twisted.web import http, resource, server
from twisted.web.util import redirectTo
def directoryListing(self):
    """
        Return a resource that generates an HTML listing of the
        directory this path represents.

        @return: A resource that renders the directory to HTML.
        @rtype: L{DirectoryLister}
        """
    path = self.path
    names = self.listNames()
    return DirectoryLister(path, names, self.contentTypes, self.contentEncodings, self.defaultType)