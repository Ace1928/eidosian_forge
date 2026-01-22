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
def make_body_seekable(self):
    """
        This forces ``environ['wsgi.input']`` to be seekable.
        That means that, the content is copied into a BytesIO or temporary
        file and flagged as seekable, so that it will not be unnecessarily
        copied again.

        After calling this method the .body_file is always seeked to the
        start of file and .content_length is not None.

        The choice to copy to BytesIO is made from
        ``self.request_body_tempfile_limit``
        """
    if self.is_body_seekable:
        self.body_file_raw.seek(0)
    else:
        self.copy_body()