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
def sanitize_url(url, remove_authority=True, remove_query_values=True, split=False):
    """
    Removes the authority and query parameter values from a given URL.
    """
    parsed_url = urlsplit(url)
    query_params = parse_qs(parsed_url.query, keep_blank_values=True)
    if remove_authority:
        netloc_parts = parsed_url.netloc.split('@')
        if len(netloc_parts) > 1:
            netloc = '%s:%s@%s' % (SENSITIVE_DATA_SUBSTITUTE, SENSITIVE_DATA_SUBSTITUTE, netloc_parts[-1])
        else:
            netloc = parsed_url.netloc
    else:
        netloc = parsed_url.netloc
    if remove_query_values:
        query_string = unquote(urlencode({key: SENSITIVE_DATA_SUBSTITUTE for key in query_params}))
    else:
        query_string = parsed_url.query
    components = Components(scheme=parsed_url.scheme, netloc=netloc, query=query_string, path=parsed_url.path, fragment=parsed_url.fragment)
    if split:
        return components
    else:
        return urlunsplit(components)