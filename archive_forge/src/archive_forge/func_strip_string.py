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
def strip_string(value, max_length=None):
    if not value:
        return value
    if max_length is None:
        max_length = DEFAULT_MAX_VALUE_LENGTH
    byte_size = _get_size_in_bytes(value)
    text_size = None
    if isinstance(value, text_type):
        text_size = len(value)
    if byte_size is not None and byte_size > max_length:
        truncated_value = _truncate_by_bytes(value, max_length)
    elif text_size is not None and text_size > max_length:
        truncated_value = value[:max_length - 3] + '...'
    else:
        return value
    return AnnotatedValue(value=truncated_value, metadata={'len': byte_size or text_size, 'rem': [['!limit', 'x', max_length - 3, max_length]]})