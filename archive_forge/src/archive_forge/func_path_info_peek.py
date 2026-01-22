import binascii
import io
import os
import re
import sys
import tempfile
import mimetypes
import warnings
from webob.acceptparse import (
from webob.cachecontrol import (
from webob.compat import (
from webob.cookies import RequestCookies
from webob.descriptors import (
from webob.etag import (
from webob.headers import EnvironHeaders
from webob.multidict import (
def path_info_peek(self):
    """
        Returns the next segment on PATH_INFO, or None if there is no
        next segment.  Doesn't modify the environment.
        """
    path = self.path_info
    if not path:
        return None
    path = path.lstrip('/')
    return path.split('/', 1)[0]