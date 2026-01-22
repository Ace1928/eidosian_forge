import base64
import json
import linecache
import logging
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections import namedtuple
from copy import copy
from decimal import Decimal
from numbers import Real
from datetime import datetime
from functools import partial
import sentry_sdk
from sentry_sdk._compat import PY2, PY33, PY37, implements_str, text_type, urlparse
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import DEFAULT_MAX_VALUE_LENGTH
def from_base64(base64_string):
    """
    Convert a string from base64, via UTF-8. Returns None on invalid input.
    """
    utf8_string = None
    try:
        only_valid_chars = BASE64_ALPHABET.match(base64_string)
        assert only_valid_chars
        base64_bytes = base64_string.encode('UTF-8')
        utf8_bytes = base64.b64decode(base64_bytes)
        utf8_string = utf8_bytes.decode('UTF-8')
    except Exception as err:
        logger.warning('Unable to decode {b64} from base64:'.format(b64=base64_string), err)
    return utf8_string