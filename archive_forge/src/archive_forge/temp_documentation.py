import os
import tempfile
from django.core.files.utils import FileProxyMixin

        Temporary file object constructor that supports reopening of the
        temporary file in Windows.

        Unlike tempfile.NamedTemporaryFile from the standard library,
        __init__() doesn't support the 'delete', 'buffering', 'encoding', or
        'newline' keyword arguments.
        