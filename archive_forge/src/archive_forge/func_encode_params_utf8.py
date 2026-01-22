from __future__ import absolute_import, unicode_literals
import collections
import datetime
import logging
import re
import sys
import time
def encode_params_utf8(params):
    """Ensures that all parameters in a list of 2-element tuples are encoded to

    bytestrings using UTF-8
    """
    encoded = []
    for k, v in params:
        encoded.append((k.encode('utf-8') if isinstance(k, unicode_type) else k, v.encode('utf-8') if isinstance(v, unicode_type) else v))
    return encoded