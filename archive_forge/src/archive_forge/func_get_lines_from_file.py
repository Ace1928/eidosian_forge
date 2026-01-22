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
def get_lines_from_file(filename, lineno, max_length=None, loader=None, module=None):
    context_lines = 5
    source = None
    if loader is not None and hasattr(loader, 'get_source'):
        try:
            source_str = loader.get_source(module)
        except (ImportError, IOError):
            source_str = None
        if source_str is not None:
            source = source_str.splitlines()
    if source is None:
        try:
            source = linecache.getlines(filename)
        except (OSError, IOError):
            return ([], None, [])
    if not source:
        return ([], None, [])
    lower_bound = max(0, lineno - context_lines)
    upper_bound = min(lineno + 1 + context_lines, len(source))
    try:
        pre_context = [strip_string(line.strip('\r\n'), max_length=max_length) for line in source[lower_bound:lineno]]
        context_line = strip_string(source[lineno].strip('\r\n'), max_length=max_length)
        post_context = [strip_string(line.strip('\r\n'), max_length=max_length) for line in source[lineno + 1:upper_bound]]
        return (pre_context, context_line, post_context)
    except IndexError:
        return ([], None, [])