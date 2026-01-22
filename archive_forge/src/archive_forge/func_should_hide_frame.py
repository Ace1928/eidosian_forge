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
def should_hide_frame(frame):
    try:
        mod = frame.f_globals['__name__']
        if mod.startswith('sentry_sdk.'):
            return True
    except (AttributeError, KeyError):
        pass
    for flag_name in ('__traceback_hide__', '__tracebackhide__'):
        try:
            if frame.f_locals[flag_name]:
                return True
        except Exception:
            pass
    return False