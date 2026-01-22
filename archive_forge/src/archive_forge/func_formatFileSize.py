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
def formatFileSize(size):
    """
    Format the given file size in bytes to human readable format.
    """
    if size < 1024:
        return '%iB' % size
    elif size < 1024 ** 2:
        return '%iK' % (size / 1024)
    elif size < 1024 ** 3:
        return '%iM' % (size / 1024 ** 2)
    else:
        return '%iG' % (size / 1024 ** 3)