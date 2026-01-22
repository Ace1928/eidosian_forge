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
def current_stacktrace(include_local_variables=True, include_source_context=True, max_value_length=None):
    __tracebackhide__ = True
    frames = []
    f = sys._getframe()
    while f is not None:
        if not should_hide_frame(f):
            frames.append(serialize_frame(f, include_local_variables=include_local_variables, include_source_context=include_source_context, max_value_length=max_value_length))
        f = f.f_back
    frames.reverse()
    return {'frames': frames}