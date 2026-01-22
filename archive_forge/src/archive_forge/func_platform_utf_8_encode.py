import json
import os
import sys
from binascii import a2b_base64
from mimetypes import guess_extension
from textwrap import dedent
from traitlets import Set, Unicode
from .base import Preprocessor
def platform_utf_8_encode(data):
    """Encode data based on platform."""
    if isinstance(data, str):
        if sys.platform == 'win32':
            data = data.replace('\n', '\r\n')
        data = data.encode('utf-8')
    return data