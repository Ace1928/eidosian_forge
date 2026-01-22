import base64
import datetime
import json
import time
import warnings
import zlib
from django.conf import settings
from django.utils.crypto import constant_time_compare, salted_hmac
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.encoding import force_bytes
from django.utils.module_loading import import_string
from django.utils.regex_helper import _lazy_re_compile
def sign_object(self, obj, serializer=JSONSerializer, compress=False):
    """
        Return URL-safe, hmac signed base64 compressed JSON string.

        If compress is True (not the default), check if compressing using zlib
        can save some space. Prepend a '.' to signify compression. This is
        included in the signature, to protect against zip bombs.

        The serializer is expected to return a bytestring.
        """
    data = serializer().dumps(obj)
    is_compressed = False
    if compress:
        compressed = zlib.compress(data)
        if len(compressed) < len(data) - 1:
            data = compressed
            is_compressed = True
    base64d = b64_encode(data).decode()
    if is_compressed:
        base64d = '.' + base64d
    return self.sign(base64d)